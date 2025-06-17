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

##############################################################################
# Memory and Performance Measurement Functions
##############################################################################

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_tensor_memory_usage():
    """Get GPU memory usage if CUDA is available, otherwise return 0"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0

def calculate_reparam_model_size(model, consider_quantization=False):
    """Calculate the memory size of ReparamModule parameters in MB"""
    total_size = 0
   
    # Handle flat_w parameter
    if hasattr(model, 'flat_w') and model.flat_w is not None:
        if consider_quantization and hasattr(model, 'is_quantized') and model.is_quantized:
            # For quantized flat_w: 1 bit per parameter + scale
            total_size += (model.flat_w.numel() / 8) + 4  # 1 bit per param + 4 bytes for scale
        else:
            if model.flat_w.dtype == torch.float32:
                total_size += model.flat_w.numel() * 4  # 4 bytes per float32
            elif model.flat_w.dtype == torch.float16:
                total_size += model.flat_w.numel() * 2  # 2 bytes per float16
            else:
                total_size += model.flat_w.numel() * 4  # Default to 4 bytes
   
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
   
    return total_size / 1024 / 1024  # Convert to MB

def log_reparam_quantization_metrics(model_id, metrics_dict, operation="quantization"):
    """
    Log ReparamModule quantization metrics in a structured format
    """
    print(f"\n📋 REPARAM MODULE {operation.upper()} METRICS LOG - MODEL {model_id}:")
    print("=" * 80)
   
    # Core measurements
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
   
    # Additional analysis
    if f'memory_change_percentage' in metrics_dict:
        print(f"\nADDITIONAL ANALYSIS:")
        print(f"  Memory change percentage: {metrics_dict['memory_change_percentage']:.2f}%")
        print(f"  Model size change percentage: {metrics_dict['model_size_change_percentage']:.2f}%")
        if 'flat_w_params' in metrics_dict:
            print(f"  Flat_w parameters: {metrics_dict['flat_w_params']:,}")
            print(f"  Average bit-width: {metrics_dict.get('average_bit_width', 16):.4f} bits")
   
    print("=" * 80)

def export_reparam_metrics_to_csv(model_id, metrics_dict, operation, filename="reparam_quantization_metrics.csv"):
    """
    Export ReparamModule metrics to CSV file for further analysis
    """
    import csv
    import os
   
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
   
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'model_id',
            'operation',
            f'memory_before_{operation}',
            f'tensor_memory_before_{operation}',
            f'model_size_before_{operation}',
            f'time_taken_for_{operation}',
            f'memory_after_{operation}',
            f'tensor_memory_after_{operation}',
            f'model_size_after_{operation}',
            f'memory_change_from_{operation}',
            f'model_size_change',
            'memory_change_percentage',
            'model_size_change_percentage',
            'flat_w_params',
            'average_bit_width'
        ]
       
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
       
        # Prepare row data
        row_data = {
            'model_id': model_id,
            'operation': operation
        }
        row_data.update(metrics_dict)
       
        writer.writerow(row_data)
   
    print(f"📊 ReparamModule metrics exported to {filename}")

### basic functions for models
def init_weights(net):
    # init_type, init_param = state.init, state.init_param
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

##############################################################################
# Enhanced Quantization Functions for ReparamModule
##############################################################################

def quantize_flat_w(flat_w):
    """1-bit quantization of flat_w tensor."""
    scale = torch.mean(torch.abs(flat_w))
    q_tensor = torch.sign(flat_w)
    return q_tensor, scale

def dequantize_flat_w(q_tensor, scale):
    """Dequantize a 1-bit quantized flat_w tensor."""
    return q_tensor * scale

##############################################################################
# ReparamModule
##############################################################################

# Helper function to initialize BatchNorm layers
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

# Initialization function for modules
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
       
        # Initialize quantization state
        net.is_quantized = False
        net.quantization_scale = None
        net.model_id = getattr(net, 'model_id', 'unknown')

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

    def quantize_flat_weights_comprehensive(self):
        """
        Comprehensive quantization of flat_w with all required measurements
        """
        print(f"\n🔧 STARTING REPARAM MODULE QUANTIZATION")
        print(f"{'='*70}")
       
        if self.is_quantized:
            print(f"Warning: Model {self.model_id} is already quantized!")
            return 0
       
        # ============ BEFORE QUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE QUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory before quantization
        memory_before_quantization = get_memory_usage()
        tensor_memory_before_quantization = get_tensor_memory_usage()
        model_size_before_quantization = calculate_reparam_model_size(self, consider_quantization=False)
       
        print(f"ReparamModule {self.model_id}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size before quantization: {model_size_before_quantization:.2f} MB")
       
        # ============ QUANTIZATION PHASE ============
        print(f"\n⚡ QUANTIZATION PHASE:")
       
        quantization_start_time = time.time()
       
        # Quantize flat_w
        q_flat_w, scale = quantize_flat_w(self.flat_w.data)
       
        # Store original flat_w and replace with quantized version
        self.original_flat_w = self.flat_w.data.clone()
        self.flat_w.data = q_flat_w
        self.quantization_scale = scale
        self.is_quantized = True
        self.flat_w.requires_grad = False  # Disable gradients for quantized weights
       
        quantization_end_time = time.time()
        quantization_time = quantization_end_time - quantization_start_time
       
        print(f"ReparamModule {self.model_id}: Time taken for quantization: {quantization_time:.4f} seconds")
       
        # ============ AFTER QUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER QUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection after quantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory after quantization
        memory_after_quantization = get_memory_usage()
        tensor_memory_after_quantization = get_tensor_memory_usage()
        model_size_after_quantization = calculate_reparam_model_size(self, consider_quantization=True)
       
        print(f"ReparamModule {self.model_id}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size after quantization: {model_size_after_quantization:.2f} MB")
       
        # ============ QUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 QUANTIZATION IMPACT ANALYSIS:")
       
        # Calculate changes
        memory_change_from_quantization = memory_before_quantization - memory_after_quantization
        tensor_memory_change = tensor_memory_before_quantization - tensor_memory_after_quantization
        model_size_reduction = model_size_before_quantization - model_size_after_quantization
       
        print(f"ReparamModule {self.model_id}: Memory reduction from quantization: {memory_change_from_quantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory change: {tensor_memory_change:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size reduction: {model_size_reduction:.2f} MB")
       
        # Calculate percentage changes
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
       
        # Calculate average bit-width
        flat_w_params = self.flat_w.numel()
        average_bit_width = 1.0  # For 1-bit quantization
       
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
            'average_bit_width': average_bit_width
        }
       
        # ============ STRUCTURED LOGGING ============
        log_reparam_quantization_metrics(self.model_id, metrics_dict, "quantization")
       
        # Export to CSV (optional - can be enabled/disabled)
        try:
            export_reparam_metrics_to_csv(self.model_id, metrics_dict, "quantization")
        except Exception as e:
            print(f"Warning: Could not export ReparamModule quantization metrics to CSV: {e}")
       
        print(f"{'='*70}")
       
        return quantization_time

    def dequantize_flat_weights_comprehensive(self):
        """
        Comprehensive dequantization of flat_w with all required measurements
        """
        print(f"\n🔧 STARTING REPARAM MODULE DEQUANTIZATION")
        print(f"{'='*70}")
       
        if not self.is_quantized:
            print(f"Warning: Model {self.model_id} is not quantized!")
            return 0
       
        # ============ BEFORE DEQUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE DEQUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory before dequantization
        memory_before_dequantization = get_memory_usage()
        tensor_memory_before_dequantization = get_tensor_memory_usage()
        model_size_before_dequantization = calculate_reparam_model_size(self, consider_quantization=True)
       
        print(f"ReparamModule {self.model_id}: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory before dequantization: {tensor_memory_before_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size before dequantization: {model_size_before_dequantization:.2f} MB")
       
        # ============ DEQUANTIZATION PHASE ============
        print(f"\n⚡ DEQUANTIZATION PHASE:")
       
        dequantization_start_time = time.time()
       
        # Dequantize flat_w
        dequantized_flat_w = dequantize_flat_w(self.flat_w.data, self.quantization_scale)
       
        # Restore dequantized flat_w
        self.flat_w.data = dequantized_flat_w
        self.is_quantized = False
        self.quantization_scale = None
        self.flat_w.requires_grad = True  # Re-enable gradients
       
        dequantization_end_time = time.time()
        dequantization_time = dequantization_end_time - dequantization_start_time
       
        print(f"ReparamModule {self.model_id}: Time taken for dequantization: {dequantization_time:.4f} seconds")
       
        # ============ AFTER DEQUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER DEQUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection after dequantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory after dequantization
        memory_after_dequantization = get_memory_usage()
        tensor_memory_after_dequantization = get_tensor_memory_usage()
        model_size_after_dequantization = calculate_reparam_model_size(self, consider_quantization=False)
       
        print(f"ReparamModule {self.model_id}: Memory usage after dequantization: {memory_after_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory after dequantization: {tensor_memory_after_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size after dequantization: {model_size_after_dequantization:.2f} MB")
       
        # ============ DEQUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 DEQUANTIZATION IMPACT ANALYSIS:")
       
        # Calculate changes
        memory_change_from_dequantization = memory_after_dequantization - memory_before_dequantization
        tensor_memory_change = tensor_memory_after_dequantization - tensor_memory_before_dequantization
        model_size_change = model_size_after_dequantization - model_size_before_dequantization
       
        print(f"ReparamModule {self.model_id}: Memory change from dequantization: {memory_change_from_dequantization:.2f} MB")
        print(f"ReparamModule {self.model_id}: Tensor memory change: {tensor_memory_change:.2f} MB")
        print(f"ReparamModule {self.model_id}: Model size change: {model_size_change:.2f} MB")
       
        # Calculate percentage changes
        memory_change_percentage = 0
        model_size_change_percentage = 0
       
        if memory_before_dequantization > 0:
            memory_change_percentage = (memory_change_from_dequantization / memory_before_dequantization) * 100
            print(f"ReparamModule {self.model_id}: Memory change percentage: {memory_change_percentage:.2f}%")
       
        if model_size_before_dequantization > 0:
            model_size_change_percentage = (model_size_change / model_size_before_dequantization) * 100
            print(f"ReparamModule {self.model_id}: Model size change percentage: {model_size_change_percentage:.2f}%")
       
        # ============ STRUCTURED METRICS COLLECTION ============
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
            'average_bit_width': 16.0  # Back to full precision
        }
       
        # ============ STRUCTURED LOGGING ============
        log_reparam_quantization_metrics(self.model_id, metrics_dict, "dequantization")
       
        # Export to CSV (optional - can be enabled/disabled)
        try:
            export_reparam_metrics_to_csv(self.model_id, metrics_dict, "dequantization")
        except Exception as e:
            print(f"Warning: Could not export ReparamModule dequantization metrics to CSV: {e}")
       
        print(f"{'='*70}")
       
        return dequantization_time

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w.to(self.flat_w.device))
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def reshape_flat_weights(self, flat_w):
        reshaped_weights = {}
        ws = flat_w.split(self._weights_numels)
        for (m, n), w, s in zip(self._weights_module_names, ws, self._weights_shapes):
            # Create a proper module name for the weight
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
        """Enhanced load_state_dict with quantization support and comprehensive measurements"""
        load_start_time = time.time()
       
        print(f"\n📥 LOADING STATE DICT FOR REPARAM MODULE {self.model_id}")
        print(f"{'='*60}")
       
        if 'flat_w' in state_dict:
            # Load the flattened weights
            self.flat_w.data = state_dict['flat_w'].detach().clone().requires_grad_(True).to(self.flat_w.device)
           
            # Handle quantization state
            if 'is_quantized' in state_dict and state_dict['is_quantized']:
                self.is_quantized = True
                self.quantization_scale = state_dict.get('quantization_scale', None)
                print(f"Model {self.model_id}: Loaded quantized state")
            else:
                self.is_quantized = False
                self.quantization_scale = None
                print(f"Model {self.model_id}: Loaded unquantized state")
           
            # Remove flat_w and quantization-related keys from state_dict for normal loading
            filtered_state_dict = {k: v for k, v in state_dict.items()
                                 if k not in ['flat_w', 'is_quantized', 'quantization_scale', 'scales', 'zero_points']}

            # Load other parameters (like BatchNorm running stats)
            if filtered_state_dict:
                super(ReparamModule, self).load_state_dict(filtered_state_dict, strict=False)

            # Unflatten the weights and set them to the modules
            ws = (t.view(s) for (t, s) in zip(self.flat_w.split(self._weights_numels), self._weights_shapes))
            for (m, n), w in zip(self._weights_module_names, ws):
                setattr(m, n, w.to(self.flat_w.device))

            # Load scales and zero_points for quantization if they exist
            if 'scales' in state_dict:
                self.scales = state_dict['scales']
            if 'zero_points' in state_dict:
                self.zero_points = state_dict['zero_points']
        else:
            # Standard state_dict loading
            super(ReparamModule, self).load_state_dict(state_dict, strict)
       
        load_end_time = time.time()
        load_time = load_end_time - load_start_time
       
        print(f"Model {self.model_id}: State dict loaded in {load_time:.4f} seconds")
        print(f"{'='*60}")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Enhanced state_dict with quantization state"""
        state = super(ReparamModule, self).state_dict(destination, prefix, keep_vars)
       
        # Add quantization state
        state[prefix + 'is_quantized'] = self.is_quantized
        if self.quantization_scale is not None:
            state[prefix + 'quantization_scale'] = self.quantization_scale
           
        return state

    def __call__(self, inp, *args, **kwargs):
        return self.forward_with_param(inp, self.flat_w, *args, **kwargs)
   
    def get_quantization_summary(self):
        """Get a summary of the current quantization state"""
        summary = {
            'model_id': self.model_id,
            'is_quantized': self.is_quantized,
            'flat_w_params': self.flat_w.numel(),
            'model_size_mb': calculate_reparam_model_size(self, consider_quantization=self.is_quantized),
            'average_bit_width': 1.0 if self.is_quantized else 16.0
        }
       
        if self.is_quantized and self.quantization_scale is not None:
            summary['quantization_scale'] = float(self.quantization_scale)
       
        return summary
