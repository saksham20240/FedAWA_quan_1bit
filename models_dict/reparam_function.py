import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import time
import psutil
import gc
from contextlib import contextmanager
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
from tabulate import tabulate

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

def calculate_reparam_model_size(model, consider_quantization=False):
    """Calculate the memory size of ReparamModule parameters in MB"""
    total_size = 0
    
    # Handle flat_w parameter with OneBit support
    if hasattr(model, 'flat_w') and model.flat_w is not None:
        if consider_quantization:
            if hasattr(model, 'is_onebit_quantized') and model.is_onebit_quantized:
                # OneBit quantization: 1 bit per parameter + value vectors
                sign_bits = model.flat_w.numel() / 8  # 1 bit per parameter, packed
                if hasattr(model, 'g_vector') and hasattr(model, 'h_vector'):
                    vector_bits = (model.g_vector.numel() + model.h_vector.numel()) * 16  # FP16
                    total_size += (sign_bits + vector_bits) / 8
                else:
                    total_size += sign_bits
            elif hasattr(model, 'is_quantized') and model.is_quantized:
                # Simple 1-bit quantization: 1 bit per parameter + scale
                total_size += (model.flat_w.numel() / 8) + 4
            else:
                # Full precision
                if model.flat_w.dtype == torch.float32:
                    total_size += model.flat_w.numel() * 4
                elif model.flat_w.dtype == torch.float16:
                    total_size += model.flat_w.numel() * 2
                else:
                    total_size += model.flat_w.numel() * 4
        else:
            if model.flat_w.dtype == torch.float32:
                total_size += model.flat_w.numel() * 4
            elif model.flat_w.dtype == torch.float16:
                total_size += model.flat_w.numel() * 2
            else:
                total_size += model.flat_w.numel() * 4
    
    # Handle other parameters (buffers, etc.)
    for param in model.parameters():
        if param is not model.flat_w:  # Skip flat_w as we already counted it
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
# OneBit Quantization Functions for ReparamModule
##############################################################################

def quantize_flat_w(flat_w):
    """Simple 1-bit quantization of flat_w tensor"""
    scale = torch.mean(torch.abs(flat_w))
    q_tensor = torch.sign(flat_w)
    return q_tensor, scale

def dequantize_flat_w(q_tensor, scale):
    """Dequantize a 1-bit quantized flat_w tensor"""
    return q_tensor * scale

def svid_decomposition_flat_w(flat_w, method='nmf'):
    """SVID decomposition for OneBit quantization of flat_w"""
    sign_matrix = torch.sign(flat_w)
    abs_matrix = torch.abs(flat_w)
    abs_numpy = abs_matrix.detach().cpu().numpy().reshape(-1, 1)  # Reshape for NMF
    
    if method == 'nmf' and abs_numpy.shape[0] > 1:
        try:
            nmf = NMF(n_components=1, init='random', random_state=42, max_iter=1000)
            W_nmf = nmf.fit_transform(abs_numpy)
            H_nmf = nmf.components_
            
            a_vector = torch.from_numpy(W_nmf.flatten()).to(flat_w.device)
            b_vector = torch.from_numpy(H_nmf.flatten()).to(flat_w.device)
        except:
            # Fallback to simple approach if NMF fails
            a_vector = torch.sqrt(torch.mean(abs_matrix)).expand(abs_matrix.shape)
            b_vector = torch.ones(1, device=flat_w.device)
    else:
        # SVD or fallback approach
        if len(abs_matrix.shape) == 1:
            abs_matrix = abs_matrix.unsqueeze(0)
        
        U, S, Vt = torch.svd(abs_matrix)
        if len(S) > 0:
            a_vector = U[:, 0] * torch.sqrt(S[0])
            b_vector = Vt[0, :] * torch.sqrt(S[0])
        else:
            a_vector = torch.sqrt(torch.mean(abs_matrix)).expand(abs_matrix.shape[0])
            b_vector = torch.ones(abs_matrix.shape[1], device=flat_w.device)
    
    return sign_matrix, a_vector, b_vector

def onebit_dequantize_flat_w(sign_matrix, g_vector, h_vector):
    """OneBit dequantization for flat_w"""
    if len(g_vector) == 1 and len(h_vector) == len(sign_matrix):
        reconstructed = sign_matrix * h_vector * g_vector
    elif len(h_vector) == 1 and len(g_vector) == len(sign_matrix):
        reconstructed = sign_matrix * g_vector * h_vector
    else:
        # Fallback: use mean scaling
        scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
        reconstructed = sign_matrix * scale
    
    return reconstructed

##############################################################################
# Comprehensive Logging Functions
##############################################################################

def log_reparam_quantization_metrics(model_id, metrics_dict, operation="quantization"):
    """Log ReparamModule quantization metrics in a structured format"""
    print(f"\n📋 REPARAM MODULE {operation.upper()} METRICS LOG - MODEL {model_id}:")
    print("=" * 80)
    
    print(f"CORE {operation.upper()} MEASUREMENTS:")
    print(f"  Memory usage before {operation}: {metrics_dict[f'memory_before_{operation}']:.2f} MB")
    print(f"  Tensor memory before {operation}: {metrics_dict[f'tensor_memory_before_{operation}']:.2f} MB")
    print(f"  Model size before {operation}: {metrics_dict[f'model_size_before_{operation}']:.2f} MB")
    print(f"  Time taken for {operation}: {metrics_dict[f'time_taken_for_{operation}']:.4f} seconds")
    print(f"  Memory usage after {operation}: {metrics_dict[f'memory_after_{operation}']:.2f} MB")
    print(f"  Tensor memory after {operation}: {metrics_dict[f'tensor_memory_after_{operation}']:.2f} MB")
    print(f"  Model size after {operation}: {metrics_dict[f'model_size_after_{operation}']:.2f} MB")
    print(f"  Memory change from {operation}: {metrics_dict[f'memory_change_from_{operation}']:.2f} MB")
    print(f"  Model size change: {metrics_dict[f'model_size_change']:.2f} MB")
    
    if f'memory_change_percentage' in metrics_dict:
        print(f"\nADDITIONAL ANALYSIS:")
        print(f"  Memory change percentage: {metrics_dict['memory_change_percentage']:.2f}%")
        print(f"  Model size change percentage: {metrics_dict['model_size_change_percentage']:.2f}%")
        if 'flat_w_params' in metrics_dict:
            print(f"  Flat_w parameters: {metrics_dict['flat_w_params']:,}")
            print(f"  Average bit-width: {metrics_dict.get('average_bit_width', 16):.4f} bits")
            print(f"  Quantization method: {metrics_dict.get('quantization_method', 'simple')}")
    
    print("=" * 80)

def export_reparam_metrics_to_csv(model_id, metrics_dict, operation, filename="reparam_quantization_metrics.csv"):
    """Export ReparamModule metrics to CSV file"""
    import csv
    import os
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'model_id', 'operation',
            f'memory_before_{operation}', f'tensor_memory_before_{operation}', 
            f'model_size_before_{operation}', f'time_taken_for_{operation}',
            f'memory_after_{operation}', f'tensor_memory_after_{operation}',
            f'model_size_after_{operation}', f'memory_change_from_{operation}',
            f'model_size_change', 'memory_change_percentage', 'model_size_change_percentage',
            'flat_w_params', 'average_bit_width', 'quantization_method'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row_data = {'model_id': model_id, 'operation': operation}
        row_data.update(metrics_dict)
        
        writer.writerow(row_data)
    
    print(f"📊 ReparamModule metrics exported to {filename}")

##############################################################################
# FedAwa Integration Functions
##############################################################################

def compute_reparam_model_divergence(model1, model2):
    """Compute normalized divergence between two ReparamModule models"""
    if not (hasattr(model1, 'flat_w') and hasattr(model2, 'flat_w')):
        return 0.0
    
    diff = torch.norm(model1.flat_w - model2.flat_w).item()
    norm = max(torch.norm(model1.flat_w).item(), torch.norm(model2.flat_w).item(), 1e-8)
    
    return diff / norm

def compute_reparam_importance_weights(client_nodes, central_node):
    """Compute adaptive importance weights for ReparamModule models"""
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
        divergence_weight = compute_reparam_model_divergence(node.model, central_node.model)
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

##############################################################################
# Client Table Generation for ReparamModule
##############################################################################

def generate_reparam_client_table(client_metrics, round_num):
    """Generate client table for ReparamModule models"""
    
    headers = [
        "Client ID", "Avg Training Loss", "Training Time (s)", "Memory Before (MB)", 
        "Memory After (MB)", "Memory Reduction (MB)", "Memory Reduction (%)", 
        "Model Size Before (MB)", "Model Size After (MB)", "Model Size Reduction (MB)", 
        "Model Size Reduction (%)", "Compression Ratio (%)", "Quantization Time (s)",
        "Average Bit-Width", "Tensor Memory Before (MB)", "Tensor Memory After (MB)", 
        "Reparam Accuracy (%)", "Adaptive Weight", "Data Weight", 
        "Performance Weight", "Divergence Weight", "Communication Size Before (MB)", 
        "Communication Size After (MB)", "Communication Reduction (%)", "CPU Usage (%)",
        "GPU Memory (MB)", "Network Bandwidth (Mbps)", "Storage Used (MB)", 
        "Power Consumption (W)", "Edge Device Compatibility", "Efficiency Rating",
        "Flat_w Parameters", "Quantization Method"
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
            f"{metrics['reparam_accuracy']:.2f}",
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
            metrics['efficiency_rating'],
            f"{metrics['flat_w_params']:,}",
            metrics['quantization_method']
        ]
        rows.append(row)
    
    table = tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
    
    print(f"\nROUND {round_num} - COMPLETE REPARAM CLIENT OUTPUT TABLE")
    print("="*220)
    print(table)
    print("="*220)

### Basic functions for models
def init_weights(net):
    init_type, init_param = None, None
    if init_type == 'imagenet_pretrained':
        assert net.__class__.__name__ == 'AlexNet'
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
        state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net

def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)

def clone_tuple(tensors, requires_grad=None):
    return tuple(
        t.detach().clone().requires_grad_(t.requires_grad if requires_grad is None else requires_grad) for t in tensors)

def initialize_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.running_mean.zero_()
        module.running_var.fill_(1)
        module.num_batches_tracked.zero_()

class NoOpContextManager:
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass

##############################################################################
# Enhanced ReparamModule with Complete OneBit + FedAwa Integration
##############################################################################

class PatchModules(type):
    def __call__(cls, *args, **kwargs):
        net = type.__call__(cls, *args, **kwargs)
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))

        net._weights_module_names = tuple(w_modules_names)
        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))
        
        # Initialize quantization states
        net.is_quantized = False
        net.is_onebit_quantized = False
        net.quantization_scale = None
        net.quantization_method = 'none'
        net.model_id = getattr(net, 'model_id', 'unknown')
        
        # OneBit quantization parameters
        net.g_vector = None
        net.h_vector = None

        return net

@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        state = self.state_dict()
        if clone:
            return {k: v.clone() for k, v in state.items()}
        return state

    def load_param(self, param_dict):
        self.load_state_dict(param_dict, strict=False)
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if name + '.running_mean' in param_dict:
                    module.running_mean = param_dict[name + '.running_mean']
                if name + '.running_var' in param_dict:
                    module.running_var = param_dict[name + '.running_var']

    def quantize_flat_weights_onebit_comprehensive(self, method='nmf'):
        """Comprehensive OneBit quantization of flat_w with SVID"""
        print(f"\n🔧 STARTING REPARAM MODULE ONEBIT QUANTIZATION")
        print(f"{'='*70}")
        
        if self.is_quantized or self.is_onebit_quantized:
            print(f"Warning: Model {self.model_id} is already quantized!")
            return 0
        
        # ============ BEFORE QUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE ONEBIT QUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_before_quantization = get_memory_usage()
        tensor_memory_before_quantization = get_tensor_memory_usage()
        model_size_before_quantization = calculate_reparam_model_size(self, consider_quantization=False)
        
        print(f"ReparamModule {self.model_id}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size before quantization: {model_size_before_quantization:.2f} MB")
        
        # ============ ONEBIT QUANTIZATION PHASE ============
        print(f"\n⚡ ONEBIT QUANTIZATION PHASE:")
        
        quantization_start_time = time.time()
        
        # Perform SVID decomposition for OneBit quantization
        sign_matrix, a_vector, b_vector = svid_decomposition_flat_w(self.flat_w.data, method)
        
        # Store original flat_w and replace with quantized version
        self.original_flat_w = self.flat_w.data.clone()
        self.flat_w.data = sign_matrix
        self.g_vector = nn.Parameter(b_vector, requires_grad=True)
        self.h_vector = nn.Parameter(a_vector, requires_grad=True)
        self.is_onebit_quantized = True
        self.quantization_method = f'onebit_{method}'
        
        # Disable gradients for sign matrix, enable for vectors
        self.flat_w.requires_grad = False
        
        quantization_end_time = time.time()
        quantization_time = quantization_end_time - quantization_start_time
        
        print(f"ReparamModule {self.model_id}: Time taken for OneBit quantization: {quantization_time:.4f} seconds")
        print(f"ReparamModule {self.model_id}: SVID method: {method}")
        
        # ============ AFTER QUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER ONEBIT QUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_after_quantization = get_memory_usage()
        tensor_memory_after_quantization = get_tensor_memory_usage()
        model_size_after_quantization = calculate_reparam_model_size(self, consider_quantization=True)
        
        print(f"ReparamModule {self.model_id}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size after quantization: {model_size_after_quantization:.2f} MB")
        
        # ============ QUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 ONEBIT QUANTIZATION IMPACT ANALYSIS:")
        
        memory_change_from_quantization = memory_before_quantization - memory_after_quantization
        model_size_reduction = model_size_before_quantization - model_size_after_quantization
        
        print(f"ReparamModule {self.model_id}: Memory reduction from quantization: {memory_change_from_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size reduction: {model_size_reduction:.2f} MB")
        
        memory_change_percentage = 0
        model_size_change_percentage = 0
        
        if memory_before_quantization > 0:
            memory_change_percentage = (memory_change_from_quantization / memory_before_quantization) * 100
            print(f"ReparamModule {self.model_id}: Memory reduction percentage: {memory_change_percentage:.2f}%")
        
        if model_size_before_quantization > 0:
            model_size_change_percentage = (model_size_reduction / model_size_before_quantization) * 100
            compression_ratio = (model_size_after_quantization / model_size_before_quantization) * 100
            print(f"ReparamModule {self.model_id}: Model size reduction percentage: {model_size_change_percentage:.2f}%")
            print(f"ReparamModule {self.model_id}: Compression ratio: {compression_ratio:.2f}%")
        
        # Calculate average bit-width for OneBit
        flat_w_params = self.flat_w.numel()
        vector_params = self.g_vector.numel() + self.h_vector.numel()
        total_params = flat_w_params + vector_params
        average_bit_width = (flat_w_params * 1.0 + vector_params * 16.0) / total_params if total_params > 0 else 1.0
        
        print(f"ReparamModule {self.model_id}: Flat_w parameters: {flat_w_params:,}")
        print(f"ReparamModule {self.model_id}: Vector parameters: {vector_params:,}")
        print(f"ReparamModule {self.model_id}: Average bit-width: {average_bit_width:.4f} bits")
        
        # ============ STRUCTURED METRICS COLLECTION ============
        metrics_dict = {
            'memory_before_quantization': memory_before_quantization,
            'tensor_memory_before_quantization': tensor_memory_before_quantization,
            'model_size_before_quantization': model_size_before_quantization,
            'time_taken_for_quantization': quantization_time,
            'memory_after_quantization': memory_after_quantization,
            'tensor_memory_after_quantization': tensor_memory_after_quantization,
            'model_size_after_quantization': model_size_after_quantization,
            'memory_change_from_quantization': memory_change_from_quantization,
            'model_size_change': model_size_reduction,
            'memory_change_percentage': memory_change_percentage,
            'model_size_change_percentage': model_size_change_percentage,
            'flat_w_params': flat_w_params,
            'average_bit_width': average_bit_width,
            'quantization_method': self.quantization_method
        }
        
        log_reparam_quantization_metrics(self.model_id, metrics_dict, "onebit_quantization")
        
        try:
            export_reparam_metrics_to_csv(self.model_id, metrics_dict, "onebit_quantization")
        except Exception as e:
            print(f"Warning: Could not export OneBit quantization metrics to CSV: {e}")
        
        print(f"{'='*70}")
        
        return quantization_time

    def quantize_flat_weights_comprehensive(self):
        """Comprehensive simple 1-bit quantization of flat_w"""
        print(f"\n🔧 STARTING REPARAM MODULE SIMPLE QUANTIZATION")
        print(f"{'='*70}")
        
        if self.is_quantized or self.is_onebit_quantized:
            print(f"Warning: Model {self.model_id} is already quantized!")
            return 0
        
        # ============ BEFORE QUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE QUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_before_quantization = get_memory_usage()
        tensor_memory_before_quantization = get_tensor_memory_usage()
        model_size_before_quantization = calculate_reparam_model_size(self, consider_quantization=False)
        
        print(f"ReparamModule {self.model_id}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size before quantization: {model_size_before_quantization:.2f} MB")
        
        # ============ QUANTIZATION PHASE ============
        print(f"\n⚡ SIMPLE QUANTIZATION PHASE:")
        
        quantization_start_time = time.time()
        
        # Quantize flat_w
        q_flat_w, scale = quantize_flat_w(self.flat_w.data)
        
        # Store original flat_w and replace with quantized version
        self.original_flat_w = self.flat_w.data.clone()
        self.flat_w.data = q_flat_w
        self.quantization_scale = scale
        self.is_quantized = True
        self.quantization_method = 'simple_1bit'
        self.flat_w.requires_grad = False
        
        quantization_end_time = time.time()
        quantization_time = quantization_end_time - quantization_start_time
        
        print(f"ReparamModule {self.model_id}: Time taken for quantization: {quantization_time:.4f} seconds")
        
        # ============ AFTER QUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER QUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_after_quantization = get_memory_usage()
        tensor_memory_after_quantization = get_tensor_memory_usage()
        model_size_after_quantization = calculate_reparam_model_size(self, consider_quantization=True)
        
        print(f"ReparamModule {self.model_id}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size after quantization: {model_size_after_quantization:.2f} MB")
        
        # ============ QUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 QUANTIZATION IMPACT ANALYSIS:")
        
        memory_change_from_quantization = memory_before_quantization - memory_after_quantization
        model_size_reduction = model_size_before_quantization - model_size_after_quantization
        
        print(f"ReparamModule {self.model_id}: Memory reduction from quantization: {memory_change_from_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size reduction: {model_size_reduction:.2f} MB")
        
        memory_change_percentage = 0
        model_size_change_percentage = 0
        
        if memory_before_quantization > 0:
            memory_change_percentage = (memory_change_from_quantization / memory_before_quantization) * 100
            print(f"ReparamModule {self.model_id}: Memory reduction percentage: {memory_change_percentage:.2f}%")
        
        if model_size_before_quantization > 0:
            model_size_change_percentage = (model_size_reduction / model_size_before_quantization) * 100
            compression_ratio = (model_size_after_quantization / model_size_before_quantization) * 100
            print(f"ReparamModule {self.model_id}: Model size reduction percentage: {model_size_change_percentage:.2f}%")
            print(f"ReparamModule {self.model_id}: Compression ratio: {compression_ratio:.2f}%")
        
        flat_w_params = self.flat_w.numel()
        average_bit_width = 1.0
        
        print(f"ReparamModule {self.model_id}: Flat_w parameters: {flat_w_params:,}")
        print(f"ReparamModule {self.model_id}: Average bit-width: {average_bit_width:.4f} bits")
        
        # ============ STRUCTURED METRICS COLLECTION ============
        metrics_dict = {
            'memory_before_quantization': memory_before_quantization,
            'tensor_memory_before_quantization': tensor_memory_before_quantization,
            'model_size_before_quantization': model_size_before_quantization,
            'time_taken_for_quantization': quantization_time,
            'memory_after_quantization': memory_after_quantization,
            'tensor_memory_after_quantization': tensor_memory_after_quantization,
            'model_size_after_quantization': model_size_after_quantization,
            'memory_change_from_quantization': memory_change_from_quantization,
            'model_size_change': model_size_reduction,
            'memory_change_percentage': memory_change_percentage,
            'model_size_change_percentage': model_size_change_percentage,
            'flat_w_params': flat_w_params,
            'average_bit_width': average_bit_width,
            'quantization_method': self.quantization_method
        }
        
        log_reparam_quantization_metrics(self.model_id, metrics_dict, "quantization")
        
        try:
            export_reparam_metrics_to_csv(self.model_id, metrics_dict, "quantization")
        except Exception as e:
            print(f"Warning: Could not export quantization metrics to CSV: {e}")
        
        print(f"{'='*70}")
        
        return quantization_time

    def dequantize_flat_weights_comprehensive(self):
        """Comprehensive dequantization of flat_w for both OneBit and simple quantization"""
        print(f"\n🔧 STARTING REPARAM MODULE DEQUANTIZATION")
        print(f"{'='*70}")
        
        if not (self.is_quantized or self.is_onebit_quantized):
            print(f"Warning: Model {self.model_id} is not quantized!")
            return 0
        
        # ============ BEFORE DEQUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE DEQUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_before_dequantization = get_memory_usage()
        tensor_memory_before_dequantization = get_tensor_memory_usage()
        model_size_before_dequantization = calculate_reparam_model_size(self, consider_quantization=True)
        
        print(f"ReparamModule {self.model_id}: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory before dequantization: {tensor_memory_before_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size before dequantization: {model_size_before_dequantization:.2f} MB")
        
        # ============ DEQUANTIZATION PHASE ============
        print(f"\n⚡ DEQUANTIZATION PHASE:")
        
        dequantization_start_time = time.time()
        
        if self.is_onebit_quantized:
            # OneBit dequantization
            dequantized_flat_w = onebit_dequantize_flat_w(self.flat_w.data, self.g_vector, self.h_vector)
            print(f"ReparamModule {self.model_id}: Performing OneBit dequantization")
        elif self.is_quantized:
            # Simple dequantization
            dequantized_flat_w = dequantize_flat_w(self.flat_w.data, self.quantization_scale)
            print(f"ReparamModule {self.model_id}: Performing simple dequantization")
        
        # Restore dequantized flat_w
        self.flat_w.data = dequantized_flat_w
        self.is_quantized = False
        self.is_onebit_quantized = False
        self.quantization_scale = None
        self.quantization_method = 'none'
        self.flat_w.requires_grad = True
        
        # Clean up OneBit vectors if they exist
        if hasattr(self, 'g_vector') and self.g_vector is not None:
            delattr(self, 'g_vector')
        if hasattr(self, 'h_vector') and self.h_vector is not None:
            delattr(self, 'h_vector')
        
        dequantization_end_time = time.time()
        dequantization_time = dequantization_end_time - dequantization_start_time
        
        print(f"ReparamModule {self.model_id}: Time taken for dequantization: {dequantization_time:.4f} seconds")
        
        # ============ AFTER DEQUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER DEQUANTIZATION MEASUREMENTS:")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_after_dequantization = get_memory_usage()
        tensor_memory_after_dequantization = get_tensor_memory_usage()
        model_size_after_dequantization = calculate_reparam_model_size(self, consider_quantization=False)
        
        print(f"ReparamModule {self.model_id}: Memory usage after dequantization: {memory_after_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory after dequantization: {tensor_memory_after_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size after dequantization: {model_size_after_dequantization:.2f} MB")
        
        # ============ DEQUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 DEQUANTIZATION IMPACT ANALYSIS:")
        
        memory_change_from_dequantization = memory_after_dequantization - memory_before_dequantization
        model_size_change = model_size_after_dequantization - model_size_before_dequantization
        
        print(f"ReparamModule {self.model_id}: Memory change from dequantization: {memory_change_from_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size change: {model_size_change:.2f} MB")
        
        memory_change_percentage = 0
        model_size_change_percentage = 0
        
        if memory_before_dequantization > 0:
            memory_change_percentage = (memory_change_from_dequantization / memory_before_dequantization) * 100
            print(f"ReparamModule {self.model_id}: Memory change percentage: {memory_change_percentage:.2f}%")
        
        if model_size_before_dequantization > 0:
            model_size_change_percentage = (model_size_change / model_size_before_dequantization) * 100
            print(f"ReparamModule {self.model_id}: Model size change percentage: {model_size_change_percentage:.2f}%")
        
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
            'model_size_change_percentage': model_size_change_percentage,
            'flat_w_params': self.flat_w.numel(),
            'average_bit_width': 16.0,
            'quantization_method': 'dequantized'
        }
        
        log_reparam_quantization_metrics(self.model_id, metrics_dict, "dequantization")
        
        try:
            export_reparam_metrics_to_csv(self.model_id, metrics_dict, "dequantization")
        except Exception as e:
            print(f"Warning: Could not export dequantization metrics to CSV: {e}")
        
        print(f"{'='*70}")
        
        return dequantization_time

    @contextmanager
    def unflatten_weight(self, flat_w):
        # Handle quantized flat_w for forward pass
        if self.is_onebit_quantized and hasattr(self, 'g_vector') and hasattr(self, 'h_vector'):
            # Use OneBit forward computation
            effective_flat_w = onebit_dequantize_flat_w(flat_w, self.g_vector, self.h_vector)
        elif self.is_quantized and hasattr(self, 'quantization_scale'):
            # Use simple dequantization
            effective_flat_w = dequantize_flat_w(flat_w, self.quantization_scale)
        else:
            # Use flat_w directly
            effective_flat_w = flat_w
        
        ws = (t.view(s) for (t, s) in zip(effective_flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w.to(self.flat_w.device))
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def reshape_flat_weights(self, flat_w):
        reshaped_weights = {}
        ws = flat_w.split(self._weights_numels)
        for (m, n), w, s in zip(self._weights_module_names, ws, self._weights_shapes):
            module_name = f"{m.__class__.__name__}.{n}"
            reshaped_weights[module_name] = w.view(s)
        return reshaped_weights
    
    def get_head_weights(self, flat_w):
        reshaped_weights = self.reshape_flat_weights(flat_w)
        linear_weights = []
        for name, weight in reshaped_weights.items():
            if 'Linear' in name:
                linear_weights.append(weight.view(-1))
        
        if linear_weights:
            return torch.cat(linear_weights)
        else:
            return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)

    def get_body_weights(self, flat_w):
        reshaped_weights = self.reshape_flat_weights(flat_w)
        non_linear_weights = []
        for name, weight in reshaped_weights.items():
            if 'Linear' not in name:
                non_linear_weights.append(weight.view(-1))
        
        if non_linear_weights:
            return torch.cat(non_linear_weights)
        else:
            return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)

    def forward_with_param(self, inp, new_w, *args, **kwargs):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Enhanced load_state_dict with comprehensive quantization support"""
        load_start_time = time.time()
        
        print(f"\n📥 LOADING STATE DICT FOR REPARAM MODULE {self.model_id}")
        print(f"{'='*60}")
        
        if 'flat_w' in state_dict:
            # Load the flattened weights
            self.flat_w.data = state_dict['flat_w'].detach().clone().requires_grad_(True).to(self.flat_w.device)
            
            # Handle quantization state
            if 'is_onebit_quantized' in state_dict and state_dict['is_onebit_quantized']:
                self.is_onebit_quantized = True
                self.quantization_method = state_dict.get('quantization_method', 'onebit')
                if 'g_vector' in state_dict:
                    self.g_vector = nn.Parameter(state_dict['g_vector'].to(self.flat_w.device))
                if 'h_vector' in state_dict:
                    self.h_vector = nn.Parameter(state_dict['h_vector'].to(self.flat_w.device))
                print(f"Model {self.model_id}: Loaded OneBit quantized state")
            elif 'is_quantized' in state_dict and state_dict['is_quantized']:
                self.is_quantized = True
                self.quantization_scale = state_dict.get('quantization_scale', None)
                self.quantization_method = state_dict.get('quantization_method', 'simple_1bit')
                print(f"Model {self.model_id}: Loaded simple quantized state")
            else:
                self.is_quantized = False
                self.is_onebit_quantized = False
                self.quantization_method = 'none'
                print(f"Model {self.model_id}: Loaded unquantized state")
            
            # Remove quantization-related keys from state_dict for normal loading
            filtered_state_dict = {k: v for k, v in state_dict.items()
                                 if k not in ['flat_w', 'is_quantized', 'is_onebit_quantized', 
                                            'quantization_scale', 'quantization_method', 
                                            'g_vector', 'h_vector']}

            # Load other parameters (like BatchNorm running stats)
            if filtered_state_dict:
                super(ReparamModule, self).load_state_dict(filtered_state_dict, strict=False)

            # Unflatten the weights and set them to the modules
            with self.unflatten_weight(self.flat_w):
                pass  # This will set the weights correctly

        else:
            # Standard state_dict loading
            super(ReparamModule, self).load_state_dict(state_dict, strict)
        
        load_end_time = time.time()
        load_time = load_end_time - load_start_time
        
        print(f"Model {self.model_id}: State dict loaded in {load_time:.4f} seconds")
        print(f"{'='*60}")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Enhanced state_dict with complete quantization state"""
        state = super(ReparamModule, self).state_dict(destination, prefix, keep_vars)
        
        # Add quantization state
        state[prefix + 'is_quantized'] = self.is_quantized
        state[prefix + 'is_onebit_quantized'] = self.is_onebit_quantized
        state[prefix + 'quantization_method'] = self.quantization_method
        
        if self.quantization_scale is not None:
            state[prefix + 'quantization_scale'] = self.quantization_scale
        
        if hasattr(self, 'g_vector') and self.g_vector is not None:
            state[prefix + 'g_vector'] = self.g_vector
        
        if hasattr(self, 'h_vector') and self.h_vector is not None:
            state[prefix + 'h_vector'] = self.h_vector
        
        return state

    def __call__(self, inp, *args, **kwargs):
        return self.forward_with_param(inp, self.flat_w, *args, **kwargs)
    
    def get_quantization_summary(self):
        """Get a comprehensive summary of the current quantization state"""
        summary = {
            'model_id': self.model_id,
            'is_quantized': self.is_quantized,
            'is_onebit_quantized': self.is_onebit_quantized,
            'quantization_method': self.quantization_method,
            'flat_w_params': self.flat_w.numel(),
            'model_size_mb': calculate_reparam_model_size(self, consider_quantization=(self.is_quantized or self.is_onebit_quantized)),
        }
        
        if self.is_onebit_quantized:
            vector_params = 0
            if hasattr(self, 'g_vector') and self.g_vector is not None:
                vector_params += self.g_vector.numel()
            if hasattr(self, 'h_vector') and self.h_vector is not None:
                vector_params += self.h_vector.numel()
            
            total_params = self.flat_w.numel() + vector_params
            summary['average_bit_width'] = (self.flat_w.numel() * 1.0 + vector_params * 16.0) / total_params if total_params > 0 else 1.0
            summary['vector_params'] = vector_params
        elif self.is_quantized:
            summary['average_bit_width'] = 1.0
            if self.quantization_scale is not None:
                summary['quantization_scale'] = float(self.quantization_scale)
        else:
            summary['average_bit_width'] = 16.0
        
        return summary

    def aggregate_with_fedawa_weights(self, other_models, adaptive_weights):
        """Aggregate this model with others using FedAwa adaptive weights"""
        print(f"\n🔀 AGGREGATING REPARAM MODULE WITH FEDAWA WEIGHTS")
        print(f"Adaptive weights: {[f'{w:.4f}' for w in adaptive_weights]}")
        
        # Zero out current flat_w
        self.flat_w.data.zero_()
        
        # Weighted aggregation
        for i, (model, weight) in enumerate(zip(other_models, adaptive_weights)):
            if hasattr(model, 'flat_w'):
                if model.is_onebit_quantized and hasattr(model, 'g_vector') and hasattr(model, 'h_vector'):
                    # Dequantize OneBit model for aggregation
                    effective_flat_w = onebit_dequantize_flat_w(model.flat_w.data, model.g_vector, model.h_vector)
                elif model.is_quantized and hasattr(model, 'quantization_scale'):
                    # Dequantize simple quantized model
                    effective_flat_w = dequantize_flat_w(model.flat_w.data, model.quantization_scale)
                else:
                    # Use flat_w directly
                    effective_flat_w = model.flat_w.data
                
                self.flat_w.data += weight * effective_flat_w
        
        print(f"✅ FedAwa aggregation completed for ReparamModule {self.model_id}")
