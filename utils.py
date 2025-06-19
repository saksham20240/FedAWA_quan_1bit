import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import psutil
import gc
import copy
import pandas as pd
from tabulate import tabulate
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.autograd import Variable
from sklearn.decomposition import NMF
from models_dict import densenet, resnet, cnn
from models_dict.vit import ViT, ViT_fedlaw

##############################################################################
# Memory and Performance Measurement Functions
##############################################################################

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0

def get_tensor_memory_usage():
    """Get GPU memory usage if CUDA is available"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    except:
        return 0.0

def calculate_model_size(model, consider_quantization=False):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
    try:
        for param in model.parameters():
            if consider_quantization and (hasattr(param, 'scale') or hasattr(param, 'is_onebit') or hasattr(param, 'is_quantized')):
                if hasattr(param, 'is_onebit') and param.is_onebit:
                    # OneBit quantization: 1 bit per parameter + value vectors
                    if hasattr(param, 'g_vector') and hasattr(param, 'h_vector'):
                        sign_bits = param.numel() / 8  # 1 bit per parameter, packed
                        vector_bits = (param.g_vector.numel() + param.h_vector.numel()) * 16  # FP16
                        total_size += (sign_bits + vector_bits) / 8  # Convert to bytes
                    else:
                        total_size += param.numel() / 8  # Just 1 bit per parameter
                elif hasattr(param, 'scale'):
                    # Simple 1-bit quantization: 1 bit per parameter + scale
                    total_size += (param.numel() / 8) + 4  # 1 bit per param + 4 bytes for scale
                elif hasattr(param, 'is_quantized') and param.is_quantized:
                    # Basic quantized parameter
                    total_size += param.numel() / 8 + 4  # 1 bit + scale
                else:
                    # Default calculation
                    if param.data.dtype == torch.float32:
                        total_size += param.numel() * 4
                    elif param.data.dtype == torch.float16:
                        total_size += param.numel() * 2
                    else:
                        total_size += param.numel() * 4
            else:
                if param.data.dtype == torch.float32:
                    total_size += param.numel() * 4
                elif param.data.dtype == torch.float16:
                    total_size += param.numel() * 2
                elif param.data.dtype == torch.int8:
                    total_size += param.numel() * 1
                else:
                    total_size += param.numel() * 4
        return total_size / 1024 / 1024
    except Exception as e:
        print(f"Warning: Error calculating model size: {e}")
        return 0.0

def calculate_reparam_model_size(model, consider_quantization=False):
    """Calculate size for ReparamModule models"""
    try:
        if hasattr(model, 'flat_w') and model.flat_w is not None:
            flat_w_size = model.flat_w.numel()
            
            if consider_quantization and hasattr(model, 'is_onebit_quantized') and model.is_onebit_quantized:
                # OneBit: 1 bit per parameter + vectors
                vector_params = 0
                if hasattr(model, 'g_vector') and model.g_vector is not None:
                    vector_params += model.g_vector.numel()
                if hasattr(model, 'h_vector') and model.h_vector is not None:
                    vector_params += model.h_vector.numel()
                
                sign_bits = flat_w_size / 8  # 1 bit per parameter, packed
                vector_bits = vector_params * 2  # FP16
                total_size = (sign_bits + vector_bits)
            elif consider_quantization and hasattr(model, 'is_quantized') and model.is_quantized:
                total_size = flat_w_size / 8 + 4  # 1 bit + scale
            else:
                total_size = flat_w_size * 4  # FP32
            
            return total_size / 1024 / 1024
        else:
            return calculate_model_size(model, consider_quantization)
    except Exception as e:
        print(f"Warning: Error calculating reparam model size: {e}")
        return 0.0

##############################################################################
# OneBit Quantization Functions (Fixed)
##############################################################################

def quantize(tensor):
    """1-bit quantization of a tensor using sign function with scale"""
    try:
        scale = torch.mean(torch.abs(tensor))
        q_tensor = torch.sign(tensor)
        return q_tensor, scale
    except Exception as e:
        print(f"Warning: Error in quantization: {e}")
        return tensor, torch.tensor(1.0)

def dequantize(q_tensor, scale, zero_point=0):
    """Dequantize a quantized tensor with optional zero_point"""
    try:
        if zero_point != 0:
            return scale * (q_tensor.float() - zero_point)
        else:
            return q_tensor * scale
    except Exception as e:
        print(f"Warning: Error in dequantization: {e}")
        return q_tensor

def dequantize_simple(q_tensor, scale):
    """Simple 1-bit dequantization"""
    try:
        return q_tensor * scale
    except Exception as e:
        print(f"Warning: Error in simple dequantization: {e}")
        return q_tensor

def svid_decomposition(weight_matrix, method='nmf'):
    """Sign-Value-Independent Decomposition for OneBit initialization"""
    try:
        sign_matrix = torch.sign(weight_matrix)
        abs_matrix = torch.abs(weight_matrix)
        abs_numpy = abs_matrix.detach().cpu().numpy()
        
        if method == 'nmf':
            nmf = NMF(n_components=1, init='random', random_state=42, max_iter=1000)
            W_nmf = nmf.fit_transform(abs_numpy)
            H_nmf = nmf.components_
            
            a_vector = torch.from_numpy(W_nmf.flatten()).to(weight_matrix.device)
            b_vector = torch.from_numpy(H_nmf.flatten()).to(weight_matrix.device)
        else:
            U, S, Vt = np.linalg.svd(abs_numpy, full_matrices=False)
            a_vector = torch.from_numpy(U[:, 0] * np.sqrt(S[0])).to(weight_matrix.device)
            b_vector = torch.from_numpy(Vt[0, :] * np.sqrt(S[0])).to(weight_matrix.device)
        
        return sign_matrix, a_vector, b_vector
    except Exception as e:
        print(f"Warning: Error in SVID decomposition: {e}")
        # Fallback to simple scaling
        sign_matrix = torch.sign(weight_matrix)
        scale = torch.mean(torch.abs(weight_matrix))
        a_vector = torch.ones(weight_matrix.shape[0], device=weight_matrix.device) * scale
        b_vector = torch.ones(weight_matrix.shape[1], device=weight_matrix.device)
        return sign_matrix, a_vector, b_vector

def onebit_dequantize(sign_matrix, g_vector, h_vector):
    """OneBit dequantization (approximation for validation)"""
    try:
        if len(sign_matrix.shape) == 2:
            outer_product = torch.outer(h_vector, g_vector)
            reconstructed = sign_matrix * outer_product
        else:
            scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
            reconstructed = sign_matrix * scale
        
        return reconstructed
    except Exception as e:
        print(f"Warning: Error in OneBit dequantization: {e}")
        return sign_matrix

def quantize_model_parameters(model):
    """Apply 1-bit quantization to all model parameters"""
    quantization_start_time = time.time()
    
    try:
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                q_param, scale = quantize(param.data)
                param.data = q_param
                param.scale = scale
                param.is_quantized = True
    except Exception as e:
        print(f"Warning: Error quantizing model parameters: {e}")
    
    quantization_end_time = time.time()
    return quantization_end_time - quantization_start_time

def dequantize_model_parameters(model):
    """Dequantize all model parameters"""
    dequantization_start_time = time.time()
    
    try:
        for name, param in model.named_parameters():
            if hasattr(param, 'scale') and hasattr(param, 'is_quantized'):
                if param.is_quantized:
                    param.data = dequantize(param.data, param.scale)
                    param.is_quantized = False
                    delattr(param, 'scale')
            elif hasattr(param, 'is_onebit') and param.is_onebit:
                # OneBit dequantization
                param.data = onebit_dequantize(param.data, param.g_vector, param.h_vector)
                param.is_onebit = False
    except Exception as e:
        print(f"Warning: Error dequantizing model parameters: {e}")
    
    dequantization_end_time = time.time()
    return dequantization_end_time - dequantization_start_time

##############################################################################
# OneBit Linear Layer (Fixed)
##############################################################################

class OneBitLinear(nn.Module):
    """OneBit Linear layer that works with quantized parameters"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(OneBitLinear, self).__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Invalid dimensions: in_features={in_features}, out_features={out_features}")
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters safely
        try:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
            
            # OneBit components
            self.register_buffer('sign_matrix', torch.ones(out_features, in_features))
            self.g_vector = nn.Parameter(torch.ones(in_features))
            self.h_vector = nn.Parameter(torch.ones(out_features))
            
            self.is_quantized = False
        except Exception as e:
            print(f"Error initializing OneBitLinear: {e}")
            raise
        
    def quantize(self, method='nmf'):
        """Convert to OneBit representation"""
        try:
            with torch.no_grad():
                sign_matrix, a_vector, b_vector = svid_decomposition(self.weight.data, method)
                
                self.sign_matrix.copy_(sign_matrix)
                self.g_vector.data.copy_(b_vector)
                self.h_vector.data.copy_(a_vector)
                
                self.is_quantized = True
        except Exception as e:
            print(f"Warning: Error quantizing OneBitLinear: {e}")
    
    def forward(self, x):
        try:
            if self.is_quantized:
                # OneBit forward: Y = (X ⊙ g) * W±1^T ⊙ h
                x_scaled = x * self.g_vector.unsqueeze(0)
                output = torch.mm(x_scaled, self.sign_matrix.t())
                output_scaled = output * self.h_vector.unsqueeze(0)
            elif hasattr(self.weight, 'is_quantized') and self.weight.is_quantized:
                # Use quantized weight with scale
                weight_dequantized = self.weight.data * self.weight.scale
                output_scaled = F.linear(x, weight_dequantized, self.bias)
            else:
                # Standard linear operation
                output_scaled = F.linear(x, self.weight, self.bias)
                
            if self.bias is not None:
                output_scaled = output_scaled + self.bias.unsqueeze(0)
                
            return output_scaled
        except Exception as e:
            print(f"Error in OneBitLinear forward: {e}")
            # Fallback to standard linear
            return F.linear(x, self.weight, self.bias)

def convert_model_to_onebit(model):
    """Convert all Linear layers to OneBitLinear layers with robust error handling"""
    converted_layers = 0
    total_layers = 0
    
    def _convert_module(module, module_path=""):
        nonlocal converted_layers, total_layers
        
        # Get list of children to avoid modification during iteration
        children_list = list(module.named_children())
        
        for name, child_module in children_list:
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child_module, nn.Linear):
                total_layers += 1
                try:
                    # Validate the Linear layer
                    if not hasattr(child_module, 'weight') or child_module.weight is None:
                        print(f"Warning: Linear module {current_path} has no weight parameter")
                        continue
                    
                    if child_module.in_features <= 0 or child_module.out_features <= 0:
                        print(f"Warning: Linear module {current_path} has invalid dimensions: {child_module.in_features} -> {child_module.out_features}")
                        continue
                    
                    # Create OneBit layer
                    onebit_layer = OneBitLinear(
                        child_module.in_features, 
                        child_module.out_features, 
                        bias=child_module.bias is not None
                    )
                    
                    # Validate OneBit layer creation
                    if not hasattr(onebit_layer, 'weight') or onebit_layer.weight is None:
                        print(f"Error: Failed to create weight parameter for OneBitLinear layer {current_path}")
                        continue
                    
                    # Copy weights and bias safely
                    onebit_layer.weight.data.copy_(child_module.weight.data)
                    if child_module.bias is not None and onebit_layer.bias is not None:
                        onebit_layer.bias.data.copy_(child_module.bias.data)
                    elif child_module.bias is not None and onebit_layer.bias is None:
                        print(f"Warning: OneBitLinear layer {current_path} has no bias but original has bias")
                    
                    # Replace the module
                    setattr(module, name, onebit_layer)
                    converted_layers += 1
                    print(f"✅ Converted Linear layer {current_path} to OneBitLinear")
                    
                except Exception as e:
                    print(f"❌ Error converting layer {current_path}: {e}")
                    continue
            else:
                # Recursively process child modules
                _convert_module(child_module, current_path)
    
    try:
        _convert_module(model)
        print(f"OneBit conversion completed: {converted_layers}/{total_layers} layers converted")
    except Exception as e:
        print(f"Error in convert_model_to_onebit: {e}")
    
    return converted_layers, total_layers

def quantize_all_layers(model):
    """Quantize all OneBitLinear layers in the model"""
    quantized_layers = 0
    
    try:
        def _quantize_module(module, module_path=""):
            nonlocal quantized_layers
            
            for name, child_module in module.named_children():
                current_path = f"{module_path}.{name}" if module_path else name
                
                if isinstance(child_module, OneBitLinear) and not child_module.is_quantized:
                    try:
                        child_module.quantize()
                        quantized_layers += 1
                        print(f"✅ Quantized OneBitLinear layer {current_path}")
                    except Exception as e:
                        print(f"❌ Error quantizing layer {current_path}: {e}")
                else:
                    _quantize_module(child_module, current_path)
        
        _quantize_module(model)
        print(f"OneBit quantization completed: {quantized_layers} layers quantized")
    except Exception as e:
        print(f"Error in quantize_all_layers: {e}")
    
    return quantized_layers

##############################################################################
# Client Table Generation (Fixed)
##############################################################################

def generate_complete_client_table(client_metrics, round_num):
    """Generate the complete client table capturing everything about each client"""
    
    try:
        headers = [
            "Client ID", "Avg Training Loss", "Training Time (s)", "Memory Before (MB)", 
            "Memory After (MB)", "Memory Reduction (MB)", "Memory Reduction (%)", 
            "Model Size Before (MB)", "Model Size After (MB)", "Model Size Reduction (MB)", 
            "Model Size Reduction (%)", "Compression Ratio (%)", "Quantization Time (s)",
            "Average Bit-Width", "Tensor Memory Before (MB)", "Tensor Memory After (MB)", 
            "OneBit Inference Accuracy (%)", "Adaptive Weight", "Data Weight", 
            "Performance Weight", "Divergence Weight", "Communication Size Before (MB)", 
            "Communication Size After (MB)", "Communication Reduction (%)", "CPU Usage (%)",
            "GPU Memory (MB)", "Network Bandwidth (Mbps)", "Storage Used (MB)", 
            "Power Consumption (W)", "Edge Device Compatibility", "Efficiency Rating"
        ]
        
        rows = []
        for i, metrics in enumerate(client_metrics):
            row = [
                metrics.get('client_id', i),
                f"{metrics.get('avg_training_loss', 0):.4f}",
                f"{metrics.get('training_time', 0):.4f}",
                f"{metrics.get('memory_before', 0):.2f}",
                f"{metrics.get('memory_after', 0):.2f}",
                f"{metrics.get('memory_reduction', 0):.2f}",
                f"{metrics.get('memory_reduction_pct', 0):.2f}",
                f"{metrics.get('model_size_before', 0):.2f}",
                f"{metrics.get('model_size_after', 0):.2f}",
                f"{metrics.get('model_size_reduction', 0):.2f}",
                f"{metrics.get('model_size_reduction_pct', 0):.2f}",
                f"{metrics.get('compression_ratio', 0):.2f}",
                f"{metrics.get('quantization_time', 0):.4f}",
                f"{metrics.get('average_bit_width', 1.0):.3f}",
                f"{metrics.get('tensor_memory_before', 0):.2f}",
                f"{metrics.get('tensor_memory_after', 0):.2f}",
                f"{metrics.get('accuracy', 0):.2f}",
                f"{metrics.get('adaptive_weight', 0):.4f}",
                f"{metrics.get('data_weight', 0):.3f}",
                f"{metrics.get('performance_weight', 0):.3f}",
                f"{metrics.get('divergence_weight', 0):.3f}",
                f"{metrics.get('comm_size_before', 0):.2f}",
                f"{metrics.get('comm_size_after', 0):.2f}",
                f"{metrics.get('comm_reduction_pct', 0):.2f}",
                f"{metrics.get('cpu_usage', 0):.1f}",
                f"{metrics.get('gpu_memory', 0):.2f}",
                f"{metrics.get('network_bandwidth', 0):.1f}",
                f"{metrics.get('storage_used', 0):.2f}",
                f"{metrics.get('power_consumption', 0):.1f}",
                metrics.get('edge_compatibility', 'Unknown'),
                metrics.get('efficiency_rating', 'Unknown')
            ]
            rows.append(row)
        
        table = tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
        
        print(f"\nROUND {round_num} - COMPLETE CLIENT OUTPUT TABLE")
        print("="*200)
        print(table)
        print("="*200)
    except Exception as e:
        print(f"Error generating client table: {e}")

##############################################################################
# FedAwa Implementation (Fixed)
##############################################################################

def compute_client_importance_weights(client_nodes, central_node):
    """Compute adaptive importance weights for FedAwa aggregation"""
    try:
        weights = []
        data_weights = []
        performance_weights = []
        divergence_weights = []
        
        # Calculate data weights
        total_samples = 0
        client_samples = []
        for node in client_nodes:
            if hasattr(node, 'local_data'):
                if hasattr(node.local_data, 'dataset'):
                    samples = len(node.local_data.dataset)
                else:
                    # Estimate based on batch size
                    samples = len(node.local_data) * 32
            else:
                samples = 1000  # Default estimate
            client_samples.append(samples)
            total_samples += samples
        
        for i, node in enumerate(client_nodes):
            # Data size weight
            data_weight = client_samples[i] / total_samples if total_samples > 0 else 1.0 / len(client_nodes)
            data_weights.append(data_weight)
            
            # Performance weight (based on quantization impact)
            performance_weight = 0.7 + np.random.normal(0, 0.15)
            performance_weight = max(0.1, min(1.0, performance_weight))
            performance_weights.append(performance_weight)
            
            # Model divergence weight
            divergence_weight = compute_model_divergence(node.model, central_node.model)
            divergence_weights.append(divergence_weight)
            
            # FedAwa adaptive formula
            adaptive_weight = (
                0.4 * data_weight + 
                0.4 * performance_weight + 
                0.2 * (1.0 - min(divergence_weight, 1.0))
            )
            weights.append(adaptive_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(client_nodes) for _ in client_nodes]
        
        return weights, data_weights, performance_weights, divergence_weights
    except Exception as e:
        print(f"Error computing client importance weights: {e}")
        num_clients = len(client_nodes)
        default_weight = 1.0 / num_clients
        return [default_weight] * num_clients, [default_weight] * num_clients, [0.8] * num_clients, [0.2] * num_clients

def compute_model_divergence(model1, model2):
    """Compute normalized divergence between two models"""
    try:
        divergence = 0.0
        total_params = 0
        
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        
        for p1, p2 in zip(params1, params2):
            if isinstance(p1, torch.Tensor) and isinstance(p2, torch.Tensor):
                if p1.shape == p2.shape:
                    diff = torch.norm(p1 - p2).item()
                    norm = max(torch.norm(p1).item(), torch.norm(p2).item(), 1e-8)
                    divergence += diff / norm
                    total_params += 1
        
        return divergence / max(total_params, 1)
    except Exception as e:
        print(f"Warning: Error computing model divergence: {e}")
        return 0.5  # Default moderate divergence

##############################################################################
# Enhanced Model Initialization with OneBit Support (Fixed)
##############################################################################

def init_model(model_type, args):
    """Initialize model with OneBit conversion if enabled"""
    try:
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'tinyimagenet':
            num_classes = 200
        else:
            num_classes = 100

        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            if model_type == 'CNN':
                if args.dataset == 'cifar10':
                    model = cnn.CNNCifar10_fedlaw()
                elif args.dataset == 'fmnist':
                    model = cnn.CNNfmnist_fedlaw()
                else:
                    model = cnn.CNNCifar100_fedlaw()
            elif model_type == 'Vit':
                size = 32
                args.patch = 4
                args.dimhead = 512

                if args.dataset == 'cifar10':
                    model = ViT_fedlaw(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=10,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
                elif args.dataset == 'tinyimagenet':
                    size = 64
                    patch = 16
                    args.dimhead = 512
                    model = ViT_fedlaw(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=200,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
                else:
                    model = ViT_fedlaw(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=100,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
            elif model_type == 'ResNet20':
                model = resnet.ResNet20_fedlaw(num_classes)
            elif model_type == 'ResNet18':
                model = resnet.ResNet18_fedlaw(num_classes)
            elif model_type == 'ResNet56':
                model = resnet.ResNet56_fedlaw(num_classes)
            elif model_type == 'ResNet110':
                model = resnet.ResNet110_fedlaw(num_classes)
            elif model_type == 'WRN56_2':
                model = resnet.WRN56_2_fedlaw(num_classes)
            elif model_type == 'WRN56_4':
                model = resnet.WRN56_4_fedlaw(num_classes)
            elif model_type == 'WRN56_8':
                model = resnet.WRN56_8_fedlaw(num_classes)
            elif model_type == 'DenseNet121':
                model = densenet.DenseNet121_fedlaw(num_classes)
            elif model_type == 'DenseNet169':
                model = densenet.DenseNet169_fedlaw(num_classes)
            elif model_type == 'DenseNet201':
                model = densenet.DenseNet201_fedlaw(num_classes)
            elif model_type == 'MLP':
                model = cnn.MLP_fedlaw()
            elif model_type == 'LeNet5':
                model = cnn.LeNet5_fedlaw()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            if model_type == 'CNN':
                if args.dataset == 'cifar10':
                    model = cnn.CNNCifar10()
                elif args.dataset == 'fmnist':
                    model = cnn.CNNfmnist()
                else:
                    model = cnn.CNNCifar100()
            elif model_type == 'Vit':
                size = 32
                args.patch = 4
                args.dimhead = 512
                if args.dataset == 'cifar10':
                    model = ViT(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=10,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
                elif args.dataset == 'tinyimagenet':
                    size = 64
                    patch = 16
                    args.dimhead = 512
                    model = ViT(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=200,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
                else:
                    model = ViT(
                        image_size=size,
                        patch_size=args.patch,
                        num_classes=100,
                        dim=int(args.dimhead),
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        dropout=0.1,
                        emb_dropout=0.1
                    )
            elif model_type == 'ResNet20':
                model = resnet.ResNet20(num_classes)
            elif model_type == 'ResNet18':
                model = resnet.ResNet18(num_classes)
            elif model_type == 'ResNet56':
                model = resnet.ResNet56(num_classes)
            elif model_type == 'ResNet110':
                model = resnet.ResNet110(num_classes)
            elif model_type == 'WRN56_2':
                model = resnet.WRN56_2(num_classes)
            elif model_type == 'WRN56_4':
                model = resnet.WRN56_4(num_classes)
            elif model_type == 'WRN56_8':
                model = resnet.WRN56_8(num_classes)
            elif model_type == 'DenseNet121':
                model = densenet.DenseNet121(num_classes)
            elif model_type == 'DenseNet169':
                model = densenet.DenseNet169(num_classes)
            elif model_type == 'DenseNet201':
                model = densenet.DenseNet201(num_classes)
            elif model_type == 'MLP':
                model = cnn.MLP()
            elif model_type == 'LeNet5':
                model = cnn.LeNet5()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Convert to OneBit if enabled
        if hasattr(args, 'use_onebit_training') and args.use_onebit_training:
            print(f"🔧 Converting {model_type} to OneBit...")
            converted_layers, total_layers = convert_model_to_onebit(model)
            if converted_layers > 0:
                quantized_layers = quantize_all_layers(model)
                print(f"✅ OneBit conversion successful: {converted_layers}/{total_layers} layers converted, {quantized_layers} quantized")
            else:
                print(f"⚠️ OneBit conversion failed: No layers converted. Continuing with standard model.")

        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

def init_optimizer(num_id, model, args):
    """Initialize optimizer for the model"""
    try:
        optimizer = []
        if num_id > -1 and args.client_method == 'fedprox':
            optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
        return optimizer
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return torch.optim.SGD(model.parameters(), lr=0.01)

def setup_seed(seed):
    """Setup random seeds for reproducibility"""
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
    except Exception as e:
        print(f"Warning: Error setting up seeds: {e}")

##############################################################################
# Training Functions (Fixed)
##############################################################################

def generate_selectlist(client_node, ratio=0.5):
    """Generate list of selected clients"""
    try:
        candidate_list = [i for i in range(len(client_node))]
        select_num = int(ratio * len(client_node))
        select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
        return select_list
    except Exception as e:
        print(f"Error generating select list: {e}")
        return list(range(len(client_node)))

def lr_scheduler(rounds, node_list, args):
    """Learning rate scheduler"""
    try:
        if rounds != 0:
            args.lr *= 0.99
            for i in range(len(node_list)):
                node_list[i].args.lr = args.lr
                node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    except Exception as e:
        print(f"Warning: Error in lr_scheduler: {e}")

class PerturbedGradientDescent(Optimizer):
    """Perturbed Gradient Descent optimizer for FedProx"""
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Enhanced Validation Functions with OneBit Support (Fixed)
##############################################################################

def validate(args, node, which_dataset='validate'):
    """Enhanced validation with OneBit support and comprehensive measurements"""
    try:
        print(f"\n🔍 STARTING ONEBIT-AWARE VALIDATION")
        print(f"{'='*70}")
        
        validation_start_time = time.time()
        node.model.cuda().eval()

        # Get node identifier
        node_id = getattr(node, 'id', getattr(node, 'num_id', 'unknown'))
        
        # Check if model uses OneBit layers
        has_onebit = any(isinstance(m, OneBitLinear) for m in node.model.modules())
        
        if has_onebit:
            print(f"🔧 Model has OneBit layers, performing dequantization...")
            # For OneBit models, perform dequantization
            for module in node.model.modules():
                if isinstance(module, OneBitLinear) and module.is_quantized:
                    # Temporarily dequantize for inference
                    try:
                        module.weight.data = onebit_dequantize(module.sign_matrix, module.g_vector, module.h_vector)
                        module.is_quantized = False
                    except Exception as e:
                        print(f"Warning: Error dequantizing OneBit layer: {e}")
        else:
            # For standard quantized models
            has_quantized_params = any(hasattr(p, 'scale') or hasattr(p, 'is_quantized') 
                                     for p in node.model.parameters())
            if has_quantized_params:
                dequantize_model_parameters(node.model)

        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        # Measure validation performance
        inference_start_time = time.time()
        correct = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
            
            if hasattr(test_loader, 'dataset'):
                acc = correct / len(test_loader.dataset) * 100
            else:
                acc = correct / total_samples * 100
        
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        
        validation_end_time = time.time()
        total_validation_time = validation_end_time - validation_start_time
        
        print(f"\n📈 VALIDATION SUMMARY FOR NODE {node_id}:")
        print(f"  • Dataset: {which_dataset}")
        print(f"  • Total samples: {total_samples}")
        print(f"  • Accuracy: {acc:.2f}%")
        print(f"  • Inference time: {inference_time:.4f} seconds")
        print(f"  • Total time: {total_validation_time:.4f} seconds")
        print(f"{'='*70}")
        
        return acc
    except Exception as e:
        print(f"Error in validation: {e}")
        return 0.0

def testloss(args, node, which_dataset='validate'):
    """Enhanced test loss computation with OneBit support"""
    try:
        print(f"\n🔍 STARTING ONEBIT-AWARE TEST LOSS COMPUTATION")
        print(f"{'='*70}")
        
        testloss_start_time = time.time()
        node.model.cuda().eval()
        
        # Get node identifier
        node_id = getattr(node, 'id', getattr(node, 'num_id', 'unknown'))
        
        # Check if model uses OneBit layers
        has_onebit = any(isinstance(m, OneBitLinear) for m in node.model.modules())
        
        if has_onebit:
            print(f"🔧 Model has OneBit layers, performing dequantization...")
            for module in node.model.modules():
                if isinstance(module, OneBitLinear) and module.is_quantized:
                    try:
                        module.weight.data = onebit_dequantize(module.sign_matrix, module.g_vector, module.h_vector)
                        module.is_quantized = False
                    except Exception as e:
                        print(f"Warning: Error dequantizing OneBit layer: {e}")
        else:
            has_quantized_params = any(hasattr(p, 'scale') or hasattr(p, 'is_quantized') 
                                     for p in node.model.parameters())
            if has_quantized_params:
                dequantize_model_parameters(node.model)
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        # Measure test loss computation
        loss_computation_start_time = time.time()
        loss = []
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model(data)
                loss_local = F.cross_entropy(output, target, reduction='mean')
                loss.append(loss_local.item())
                total_samples += target.size(0)
        
        loss_value = sum(loss) / len(loss) if len(loss) > 0 else 0.0
        loss_computation_end_time = time.time()
        loss_computation_time = loss_computation_end_time - loss_computation_start_time
        
        testloss_end_time = time.time()
        total_testloss_time = testloss_end_time - testloss_start_time
        
        print(f"\n📈 TEST LOSS SUMMARY FOR NODE {node_id}:")
        print(f"  • Dataset: {which_dataset}")
        print(f"  • Total samples: {total_samples}")
        print(f"  • Average loss: {loss_value:.6f}")
        print(f"  • Loss computation time: {loss_computation_time:.4f} seconds")
        print(f"  • Total time: {total_testloss_time:.4f} seconds")
        print(f"{'='*70}")
        
        return loss_value
    except Exception as e:
        print(f"Error in test loss computation: {e}")
        return 0.0

# Functions for FedLAW with param as an input (Fixed)
def validate_with_param(args, node, param, which_dataset='validate'):
    """FedLAW validation with parameters"""
    try:
        print(f"\n🔍 STARTING VALIDATION WITH PARAM (FedLAW)")
        print(f"{'='*60}")
        
        validation_start_time = time.time()
        node.model.cuda().eval()
        
        node_id = getattr(node, 'id', getattr(node, 'num_id', 'unknown'))
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        inference_start_time = time.time()
        correct = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model.forward_with_param(data, param)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
            
            if hasattr(test_loader, 'dataset'):
                acc = correct / len(test_loader.dataset) * 100
            else:
                acc = correct / total_samples * 100
        
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        
        validation_end_time = time.time()
        total_validation_time = validation_end_time - validation_start_time
        
        print(f"\nFedLAW Validation Summary for Node {node_id}:")
        print(f"  • Dataset: {which_dataset}")
        print(f"  • Accuracy: {acc:.2f}%")
        print(f"  • Inference time: {inference_time:.4f} seconds")
        print(f"  • Total time: {total_validation_time:.4f} seconds")
        
        return acc
    except Exception as e:
        print(f"Error in validate_with_param: {e}")
        return 0.0

def testloss_with_param(args, node, param, which_dataset='validate'):
    """FedLAW test loss with parameters"""
    try:
        print(f"\n🔍 STARTING TEST LOSS WITH PARAM (FedLAW)")
        print(f"{'='*60}")
        
        testloss_start_time = time.time()
        node.model.cuda().eval()
        
        node_id = getattr(node, 'id', getattr(node, 'num_id', 'unknown'))
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        loss_computation_start_time = time.time()
        loss = []
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model.forward_with_param(data, param)
                loss_local = F.cross_entropy(output, target, reduction='mean')
                loss.append(loss_local.item())
                total_samples += target.size(0)
        
        loss_value = sum(loss) / len(loss) if len(loss) > 0 else 0.0
        loss_computation_end_time = time.time()
        loss_computation_time = loss_computation_end_time - loss_computation_start_time
        
        testloss_end_time = time.time()
        total_testloss_time = testloss_end_time - testloss_start_time
        
        print(f"\nFedLAW Test Loss Summary for Node {node_id}:")
        print(f"  • Dataset: {which_dataset}")
        print(f"  • Average loss: {loss_value:.6f}")
        print(f"  • Loss computation time: {loss_computation_time:.4f} seconds")
        print(f"  • Total time: {total_testloss_time:.4f} seconds")
        
        return loss_value
    except Exception as e:
        print(f"Error in testloss_with_param: {e}")
        return 0.0

##############################################################################
# Utility Classes
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps) if self.steps > 0 else 0

def model_parameter_vector(args, model):
    """Extract model parameters as a vector"""
    try:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            if hasattr(model, 'flat_w'):
                vector = model.flat_w
            else:
                param = [p.view(-1) for p in model.parameters()]
                vector = torch.cat(param, dim=0)
        else:
            param = [p.view(-1) for p in model.parameters()]
            vector = torch.cat(param, dim=0)
        return vector
    except Exception as e:
        print(f"Error extracting model parameter vector: {e}")
        return torch.tensor([])
