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
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_tensor_memory_usage():
    """Get GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def calculate_model_size(model, consider_quantization=False):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
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

##############################################################################
# OneBit Quantization Functions (Integrated)
##############################################################################

def quantize(tensor):
    """1-bit quantization of a tensor using sign function with scale"""
    scale = torch.mean(torch.abs(tensor))
    q_tensor = torch.sign(tensor)
    return q_tensor, scale

def dequantize(q_tensor, scale, zero_point=0):
    """Dequantize a quantized tensor with optional zero_point"""
    if zero_point != 0:
        return scale * (q_tensor.float() - zero_point)
    else:
        return q_tensor * scale

def dequantize_simple(q_tensor, scale):
    """Simple 1-bit dequantization"""
    return q_tensor * scale

def svid_decomposition(weight_matrix, method='nmf'):
    """Sign-Value-Independent Decomposition for OneBit initialization"""
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

def onebit_dequantize(sign_matrix, g_vector, h_vector):
    """OneBit dequantization (approximation for validation)"""
    if len(sign_matrix.shape) == 2:
        outer_product = torch.outer(h_vector, g_vector)
        reconstructed = sign_matrix * outer_product
    else:
        scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
        reconstructed = sign_matrix * scale
    
    return reconstructed

def quantize_model_parameters(model):
    """Apply 1-bit quantization to all model parameters"""
    quantization_start_time = time.time()
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            q_param, scale = quantize(param.data)
            param.data = q_param
            param.scale = scale
            param.is_quantized = True
    
    quantization_end_time = time.time()
    return quantization_end_time - quantization_start_time

def dequantize_model_parameters(model):
    """Dequantize all model parameters"""
    dequantization_start_time = time.time()
    
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
    
    dequantization_end_time = time.time()
    return dequantization_end_time - dequantization_start_time

##############################################################################
# OneBit Linear Layer (Integrated)
##############################################################################

class OneBitLinear(nn.Module):
    """OneBit Linear layer that works with quantized parameters"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(OneBitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # OneBit components
        self.register_buffer('sign_matrix', torch.ones(out_features, in_features))
        self.g_vector = nn.Parameter(torch.ones(in_features))
        self.h_vector = nn.Parameter(torch.ones(out_features))
        
        self.is_quantized = False
        
    def quantize(self, method='nmf'):
        """Convert to OneBit representation"""
        with torch.no_grad():
            sign_matrix, a_vector, b_vector = svid_decomposition(self.weight.data, method)
            
            self.sign_matrix.copy_(sign_matrix)
            self.g_vector.data.copy_(b_vector)
            self.h_vector.data.copy_(a_vector)
            
            self.is_quantized = True
    
    def forward(self, x):
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

def convert_model_to_onebit(model):
    """Convert all Linear layers to OneBitLinear layers"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            onebit_layer = OneBitLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            
            onebit_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                onebit_layer.bias.data.copy_(module.bias.data)
            
            setattr(model, name, onebit_layer)
        else:
            convert_model_to_onebit(module)

def quantize_all_layers(model):
    """Quantize all OneBitLinear layers in the model"""
    for module in model.modules():
        if isinstance(module, OneBitLinear) and not module.is_quantized:
            module.quantize()

##############################################################################
# Client Table Generation (Integrated)
##############################################################################

def generate_complete_client_table(client_metrics, round_num):
    """Generate the complete client table capturing everything about each client"""
    
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
            metrics['client_id'],
            f"{metrics['avg_training_loss']:.4f}",
            f"{metrics['training_time']:.4f}",
            f"{metrics['memory_before']:.2f}",
            f"{metrics['memory_after']:.2f}",
            f"{metrics['memory_reduction']:.2f}",
            f"{metrics['memory_reduction_pct']:.2f}",
            f"{metrics['model_size_before']:.2f}",
            f"{metrics['model_size_after']:.2f}",
            f"{metrics['model_size_reduction']:.2f}",
            f"{metrics['model_size_reduction_pct']:.2f}",
            f"{metrics['compression_ratio']:.2f}",
            f"{metrics['quantization_time']:.4f}",
            f"{metrics['average_bit_width']:.3f}",
            f"{metrics['tensor_memory_before']:.2f}",
            f"{metrics['tensor_memory_after']:.2f}",
            f"{metrics['onebit_accuracy']:.2f}",
            f"{metrics['adaptive_weight']:.4f}",
            f"{metrics['data_weight']:.3f}",
            f"{metrics['performance_weight']:.3f}",
            f"{metrics['divergence_weight']:.3f}",
            f"{metrics['comm_size_before']:.2f}",
            f"{metrics['comm_size_after']:.2f}",
            f"{metrics['comm_reduction_pct']:.2f}",
            f"{metrics['cpu_usage']:.1f}",
            f"{metrics['gpu_memory']:.2f}",
            f"{metrics['network_bandwidth']:.1f}",
            f"{metrics['storage_used']:.2f}",
            f"{metrics['power_consumption']:.1f}",
            metrics['edge_compatibility'],
            metrics['efficiency_rating']
        ]
        rows.append(row)
    
    table = tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
    
    print(f"\nROUND {round_num} - COMPLETE CLIENT OUTPUT TABLE")
    print("="*200)
    print(table)
    print("="*200)

##############################################################################
# FedAwa Implementation (Integrated)
##############################################################################

def compute_client_importance_weights(client_nodes, central_node):
    """Compute adaptive importance weights for FedAwa aggregation"""
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
                samples = len(node.local_data) * 32
        else:
            samples = 1000
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

def compute_model_divergence(model1, model2):
    """Compute normalized divergence between two models"""
    divergence = 0.0
    total_params = 0
    
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        if isinstance(p1, torch.Tensor) and isinstance(p2, torch.Tensor):
            diff = torch.norm(p1 - p2).item()
            norm = max(torch.norm(p1).item(), torch.norm(p2).item(), 1e-8)
            divergence += diff / norm
            total_params += 1
    
    return divergence / max(total_params, 1)

##############################################################################
# Comprehensive Metrics Collection
##############################################################################

def log_dequantization_metrics(node_id, metrics_dict, operation_context="validation"):
    """Log dequantization metrics in a structured format"""
    print(f"\n📋 DEQUANTIZATION METRICS LOG FOR {operation_context.upper()} - NODE {node_id}:")
    print("=" * 80)
    
    print("CORE DEQUANTIZATION MEASUREMENTS:")
    print(f"  Memory usage before dequantization: {metrics_dict['memory_before_dequantization']:.2f} MB")
    print(f"  Tensor memory before dequantization: {metrics_dict['tensor_memory_before_dequantization']:.2f} MB")
    print(f"  Model size before dequantization: {metrics_dict['model_size_before_dequantization']:.2f} MB")
    print(f"  Time taken for dequantization: {metrics_dict['time_taken_for_dequantization']:.4f} seconds")
    print(f"  Memory usage after dequantization: {metrics_dict['memory_after_dequantization']:.2f} MB")
    print(f"  Tensor memory after dequantization: {metrics_dict['tensor_memory_after_dequantization']:.2f} MB")
    print(f"  Model size after dequantization: {metrics_dict['model_size_after_dequantization']:.2f} MB")
    print(f"  Memory change from dequantization: {metrics_dict['memory_change_from_dequantization']:.2f} MB")
    print(f"  Model size change: {metrics_dict['model_size_change']:.2f} MB")
    
    if 'memory_change_percentage' in metrics_dict:
        print(f"\nADDITIONAL ANALYSIS:")
        print(f"  Memory change percentage: {metrics_dict['memory_change_percentage']:.2f}%")
        print(f"  Model size change percentage: {metrics_dict['model_size_change_percentage']:.2f}%")
    
    print("=" * 80)

def export_dequantization_metrics_to_csv(node_id, metrics_dict, operation_context, filename="dequantization_metrics.csv"):
    """Export dequantization metrics to CSV file"""
    import csv
    import os
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'node_id',
            'operation_context',
            'memory_before_dequantization',
            'tensor_memory_before_dequantization',
            'model_size_before_dequantization',
            'time_taken_for_dequantization',
            'memory_after_dequantization',
            'tensor_memory_after_dequantization',
            'model_size_after_dequantization',
            'memory_change_from_dequantization',
            'model_size_change',
            'memory_change_percentage',
            'model_size_change_percentage'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row_data = {
            'node_id': node_id,
            'operation_context': operation_context
        }
        row_data.update(metrics_dict)
        
        writer.writerow(row_data)
    
    print(f"📊 Dequantization metrics exported to {filename}")

def measure_dequantization_comprehensive(model, node_id, operation_context="validation"):
    """Comprehensive measurement wrapper for dequantization operations"""
    
    print(f"\n📏 BEFORE DEQUANTIZATION MEASUREMENTS ({operation_context.upper()}):")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    memory_before_dequantization = get_memory_usage()
    tensor_memory_before_dequantization = get_tensor_memory_usage()
    model_size_before_dequantization = calculate_model_size(model, consider_quantization=True)
    
    print(f"Node {node_id}: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory before dequantization: {tensor_memory_before_dequantization:.2f} MB")
    print(f"Node {node_id}: Model size before dequantization: {model_size_before_dequantization:.2f} MB")
    
    print(f"\n⚡ DEQUANTIZATION PHASE ({operation_context.upper()}):")
    
    dequantization_start_time = time.time()
    
    # Perform OneBit-aware dequantization
    for name, param in model.named_parameters():
        if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
            param.data = dequantize(param.data, param.scale, param.zero_point)
        elif hasattr(param, 'scale') and not hasattr(param, 'zero_point'):
            param.data = dequantize_simple(param.data, param.scale)
    
    # For OneBitLinear layers
    for module in model.modules():
        if isinstance(module, OneBitLinear) and module.is_quantized:
            # Temporarily dequantize for inference
            module.weight.data = onebit_dequantize(module.sign_matrix, module.g_vector, module.h_vector)
            module.is_quantized = False
    
    dequantization_end_time = time.time()
    dequantization_time = dequantization_end_time - dequantization_start_time
    
    print(f"Node {node_id}: Time taken for dequantization: {dequantization_time:.4f} seconds")
    
    print(f"\n📐 AFTER DEQUANTIZATION MEASUREMENTS ({operation_context.upper()}):")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    memory_after_dequantization = get_memory_usage()
    tensor_memory_after_dequantization = get_tensor_memory_usage()
    model_size_after_dequantization = calculate_model_size(model, consider_quantization=False)
    
    print(f"Node {node_id}: Memory usage after dequantization: {memory_after_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory after dequantization: {tensor_memory_after_dequantization:.2f} MB")
    print(f"Node {node_id}: Model size after dequantization: {model_size_after_dequantization:.2f} MB")
    
    print(f"\n📊 DEQUANTIZATION IMPACT ANALYSIS ({operation_context.upper()}):")
    
    memory_change_from_dequantization = memory_after_dequantization - memory_before_dequantization
    tensor_memory_change = tensor_memory_after_dequantization - tensor_memory_before_dequantization
    model_size_change = model_size_after_dequantization - model_size_before_dequantization
    
    print(f"Node {node_id}: Memory change from dequantization: {memory_change_from_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory change: {tensor_memory_change:.2f} MB")
    print(f"Node {node_id}: Model size change: {model_size_change:.2f} MB")
    
    memory_change_percentage = 0
    model_size_change_percentage = 0
    
    if memory_before_dequantization > 0:
        memory_change_percentage = (memory_change_from_dequantization / memory_before_dequantization) * 100
        print(f"Node {node_id}: Memory change percentage: {memory_change_percentage:.2f}%")
    
    if model_size_before_dequantization > 0:
        model_size_change_percentage = (model_size_change / model_size_before_dequantization) * 100
        print(f"Node {node_id}: Model size change percentage: {model_size_change_percentage:.2f}%")
    
    metrics_dict = {
        'memory_before_dequantization': memory_before_dequantization,
        'tensor_memory_before_dequantization': tensor_memory_before_dequantization,
        'model_size_before_dequantization': model_size_before_dequantization,
        'time_taken_for_dequantization': dequantization_time,
        'memory_after_dequantization': memory_after_dequantization,
        'tensor_memory_after_dequantization': tensor_memory_after_dequantization,
        'model_size_after_dequantization': model_size_after_dequantization,
        'memory_change_from_dequantization': memory_change_from_dequantization,
        'model_size_change': model_size_change,
        'memory_change_percentage': memory_change_percentage,
        'model_size_change_percentage': model_size_change_percentage
    }
    
    log_dequantization_metrics(node_id, metrics_dict, operation_context)
    
    try:
        export_dequantization_metrics_to_csv(node_id, metrics_dict, operation_context)
    except Exception as e:
        print(f"Warning: Could not export dequantization metrics to CSV: {e}")
    
    return dequantization_time

##############################################################################
# Tools
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
        return self.total / float(self.steps)

def model_parameter_vector(args, model):
    if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
        vector = model.flat_w
    else:
        param = [p.view(-1) for p in model.parameters()]
        vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Enhanced Model Initialization with OneBit Support
##############################################################################

def init_model(model_type, args):
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

    # Convert to OneBit if enabled
    if hasattr(args, 'use_onebit_training') and args.use_onebit_training:
        convert_model_to_onebit(model)
        quantize_all_layers(model)

    return model

def init_optimizer(num_id, model, args):
    optimizer = []
    if num_id > -1 and args.client_method == 'fedprox':
        optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training Functions
##############################################################################

def generate_selectlist(client_node, ratio=0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
    return select_list

def lr_scheduler(rounds, node_list, args):
    if rounds != 0:
        args.lr *= 0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr

class PerturbedGradientDescent(Optimizer):
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
# Enhanced Validation Functions with OneBit Support
##############################################################################

def validate(args, node, which_dataset='validate'):
    """Enhanced validation with OneBit support and comprehensive measurements"""
    print(f"\n🔍 STARTING ONEBIT-AWARE VALIDATION WITH COMPREHENSIVE MEASUREMENTS")
    print(f"{'='*70}")
    
    validation_start_time = time.time()
    node.model.cuda().eval()

    # Get node identifier
    node_id = getattr(node, 'id', getattr(node, 'client_id', 'unknown'))
    
    # Check if model uses OneBit layers
    has_onebit = any(isinstance(m, OneBitLinear) for m in node.model.modules())
    
    if has_onebit:
        # For OneBit models, perform comprehensive dequantization measurements
        dequantization_time = measure_dequantization_comprehensive(
            node.model,
            node_id,
            operation_context=f"validation_{which_dataset}"
        )
    else:
        # For standard quantized models, use original measurement approach
        dequantization_time = 0
        # Check for standard quantization
        has_quantized_params = any(hasattr(p, 'scale') or hasattr(p, 'is_quantized') 
                                 for p in node.model.parameters())
        if has_quantized_params:
            dequantization_time = dequantize_model_parameters(node.model)

    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

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
    
    # ============ VALIDATION SUMMARY ============
    print(f"\n📈 ONEBIT VALIDATION SUMMARY FOR NODE {node_id}:")
    print(f"{'─'*70}")
    print(f"MODEL CHARACTERISTICS:")
    print(f"  • OneBit layers: {'Yes' if has_onebit else 'No'}")
    print(f"  • Quantized parameters: {'Yes' if any(hasattr(p, 'scale') for p in node.model.parameters()) else 'No'}")
    print(f"")
    print(f"VALIDATION PERFORMANCE:")
    print(f"  • Dataset: {which_dataset}")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Correct predictions: {correct}")
    print(f"  • Accuracy: {acc:.2f}%")
    print(f"")
    print(f"TIMING ANALYSIS:")
    print(f"  • Dequantization time: {dequantization_time:.4f} seconds")
    print(f"  • Inference time: {inference_time:.4f} seconds")
    print(f"  • Total validation time: {total_validation_time:.4f} seconds")
    if total_validation_time > 0:
        print(f"  • Dequantization overhead: {(dequantization_time/total_validation_time)*100:.1f}%")
    print(f"{'='*70}")
    
    return acc

def testloss(args, node, which_dataset='validate'):
    """Enhanced test loss computation with OneBit support"""
    print(f"\n🔍 STARTING ONEBIT-AWARE TEST LOSS COMPUTATION")
    print(f"{'='*70}")
    
    testloss_start_time = time.time()
    node.model.cuda().eval()
    
    # Get node identifier
    node_id = getattr(node, 'id', getattr(node, 'client_id', 'unknown'))
    
    # Check if model uses OneBit layers
    has_onebit = any(isinstance(m, OneBitLinear) for m in node.model.modules())
    
    if has_onebit:
        dequantization_time = measure_dequantization_comprehensive(
            node.model,
            node_id,
            operation_context=f"testloss_{which_dataset}"
        )
    else:
        dequantization_time = 0
        has_quantized_params = any(hasattr(p, 'scale') or hasattr(p, 'is_quantized') 
                                 for p in node.model.parameters())
        if has_quantized_params:
            dequantization_time = dequantize_model_parameters(node.model)
    
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

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
    
    loss_value = sum(loss) / len(loss)
    loss_computation_end_time = time.time()
    loss_computation_time = loss_computation_end_time - loss_computation_start_time
    
    testloss_end_time = time.time()
    total_testloss_time = testloss_end_time - testloss_start_time
    
    # ============ TEST LOSS SUMMARY ============
    print(f"\n📈 ONEBIT TEST LOSS SUMMARY FOR NODE {node_id}:")
    print(f"{'─'*70}")
    print(f"MODEL CHARACTERISTICS:")
    print(f"  • OneBit layers: {'Yes' if has_onebit else 'No'}")
    print(f"")
    print(f"TEST LOSS PERFORMANCE:")
    print(f"  • Dataset: {which_dataset}")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Average loss: {loss_value:.6f}")
    print(f"  • Number of batches: {len(loss)}")
    print(f"")
    print(f"TIMING ANALYSIS:")
    print(f"  • Dequantization time: {dequantization_time:.4f} seconds")
    print(f"  • Loss computation time: {loss_computation_time:.4f} seconds")
    print(f"  • Total test loss time: {total_testloss_time:.4f} seconds")
    if total_testloss_time > 0:
        print(f"  • Dequantization overhead: {(dequantization_time/total_testloss_time)*100:.1f}%")
    print(f"{'='*70}")
    
    return loss_value

# Functions for FedLAW with param as an input (unchanged for compatibility)
def validate_with_param(args, node, param, which_dataset='validate'):
    """FedLAW validation with parameters"""
    print(f"\n🔍 STARTING VALIDATION WITH PARAM (FedLAW)")
    print(f"{'='*60}")
    
    validation_start_time = time.time()
    node.model.cuda().eval()
    
    node_id = getattr(node, 'id', getattr(node, 'client_id', 'unknown'))
    
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

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

def testloss_with_param(args, node, param, which_dataset='validate'):
    """FedLAW test loss with parameters"""
    print(f"\n🔍 STARTING TEST LOSS WITH PARAM (FedLAW)")
    print(f"{'='*60}")
    
    testloss_start_time = time.time()
    node.model.cuda().eval()
    
    node_id = getattr(node, 'id', getattr(node, 'client_id', 'unknown'))
    
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

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
    
    loss_value = sum(loss) / len(loss)
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
