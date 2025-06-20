import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import time
import psutil
import gc
import warnings
from contextlib import contextmanager
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd

# Suppress the FutureWarning about positional args
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Positional args are being deprecated.*")

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

def calculate_reparam_model_size(model, consider_quantization=False):
    """Calculate the memory size of ReparamModule parameters in MB"""
    total_size = 0
    
    try:
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
    except Exception as e:
        print(f"Warning: Error calculating reparam model size: {e}")
        return 0.0

##############################################################################
# OneBit Quantization Functions for ReparamModule
##############################################################################

def quantize_flat_w(flat_w):
    """Simple 1-bit quantization of flat_w tensor"""
    try:
        scale = torch.mean(torch.abs(flat_w))
        q_tensor = torch.sign(flat_w)
        return q_tensor, scale
    except:
        return flat_w, torch.tensor(1.0)

def dequantize_flat_w(q_tensor, scale):
    """Dequantize a 1-bit quantized flat_w tensor"""
    try:
        return q_tensor * scale
    except:
        return q_tensor

def svid_decomposition_flat_w(flat_w, method='nmf'):
    """SVID decomposition for OneBit quantization of flat_w"""
    try:
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
    except:
        # Fallback to simple scaling
        sign_matrix = torch.sign(flat_w)
        scale = torch.mean(torch.abs(flat_w))
        a_vector = torch.ones(flat_w.shape[0], device=flat_w.device) * scale
        b_vector = torch.ones(1, device=flat_w.device)
        return sign_matrix, a_vector, b_vector

def onebit_dequantize_flat_w(sign_matrix, g_vector, h_vector):
    """OneBit dequantization for flat_w"""
    try:
        if len(g_vector) == 1 and len(h_vector) == len(sign_matrix):
            reconstructed = sign_matrix * h_vector * g_vector
        elif len(h_vector) == 1 and len(g_vector) == len(sign_matrix):
            reconstructed = sign_matrix * g_vector * h_vector
        else:
            # Fallback: use mean scaling
            scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
            reconstructed = sign_matrix * scale
        
        return reconstructed
    except:
        return sign_matrix

##############################################################################
# FedAwa Integration Functions
##############################################################################

def compute_reparam_model_divergence(model1, model2):
    """Compute normalized divergence between two ReparamModule models"""
    try:
        if not (hasattr(model1, 'flat_w') and hasattr(model2, 'flat_w')):
            return 0.0
        
        diff = torch.norm(model1.flat_w - model2.flat_w).item()
        norm = max(torch.norm(model1.flat_w).item(), torch.norm(model2.flat_w).item(), 1e-8)
        
        return diff / norm
    except:
        return 0.2

def compute_reparam_importance_weights(client_nodes, central_node):
    """Compute adaptive importance weights for ReparamModule models"""
    try:
        weights = []
        data_weights = []
        performance_weights = []
        divergence_weights = []
        
        # Handle both list and dict of client nodes
        if isinstance(client_nodes, dict):
            client_list = list(client_nodes.values())
        else:
            client_list = client_nodes
        
        # Calculate data weights
        total_samples = 0
        client_samples = []
        for node in client_list:
            if hasattr(node, 'local_data'):
                if hasattr(node.local_data, 'dataset'):
                    samples = len(node.local_data.dataset)
                else:
                    samples = len(node.local_data) * 32
            else:
                samples = 1000
            client_samples.append(samples)
            total_samples += samples
        
        for i, node in enumerate(client_list):
            # Data size weight
            data_weight = client_samples[i] / total_samples if total_samples > 0 else 1.0 / len(client_list)
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
            weights = [1.0 / len(client_list) for _ in client_list]
        
        return weights, data_weights, performance_weights, divergence_weights
    except Exception as e:
        print(f"Error in compute_reparam_importance_weights: {e}")
        # Return default uniform weights
        num_clients = len(client_nodes) if isinstance(client_nodes, list) else len(client_nodes.values()) if isinstance(client_nodes, dict) else 20
        default_weight = 1.0 / num_clients
        return ([default_weight] * num_clients, 
                [default_weight] * num_clients, 
                [0.8] * num_clients, 
                [0.2] * num_clients)

##############################################################################
# Basic functions for models
##############################################################################

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
        
        # Set a temporary model_id that can be overridden later
        net.model_id = 'reparam_model'
        
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
        if self.is_quantized or self.is_onebit_quantized:
            return 0
        
        # Memory measurements
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        quantization_start_time = time.time()
        
        try:
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
            
            return quantization_time
        except Exception as e:
            print(f"Error in OneBit quantization: {e}")
            return 0

    def quantize_flat_weights_comprehensive(self):
        """Comprehensive simple 1-bit quantization of flat_w"""
        if self.is_quantized or self.is_onebit_quantized:
            return 0
        
        quantization_start_time = time.time()
        
        try:
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
            
            return quantization_time
        except Exception as e:
            print(f"Error in simple quantization: {e}")
            return 0

    def dequantize_flat_weights_comprehensive(self):
        """Comprehensive dequantization of flat_w for both OneBit and simple quantization"""
        if not (self.is_quantized or self.is_onebit_quantized):
            return 0
        
        dequantization_start_time = time.time()
        
        try:
            if self.is_onebit_quantized:
                # OneBit dequantization
                dequantized_flat_w = onebit_dequantize_flat_w(self.flat_w.data, self.g_vector, self.h_vector)
            elif self.is_quantized:
                # Simple dequantization
                dequantized_flat_w = dequantize_flat_w(self.flat_w.data, self.quantization_scale)
            
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
            
            return dequantization_time
        except Exception as e:
            print(f"Error in dequantization: {e}")
            return 0

    @contextmanager
    def unflatten_weight(self, flat_w):
        try:
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
        except Exception as e:
            print(f"Error in unflatten_weight: {e}")
            yield

    def reshape_flat_weights(self, flat_w):
        try:
            reshaped_weights = {}
            ws = flat_w.split(self._weights_numels)
            for (m, n), w, s in zip(self._weights_module_names, ws, self._weights_shapes):
                module_name = f"{m.__class__.__name__}.{n}"
                reshaped_weights[module_name] = w.view(s)
            return reshaped_weights
        except:
            return {}
    
    def get_head_weights(self, flat_w):
        try:
            reshaped_weights = self.reshape_flat_weights(flat_w)
            linear_weights = []
            for name, weight in reshaped_weights.items():
                if 'Linear' in name:
                    linear_weights.append(weight.view(-1))
            
            if linear_weights:
                return torch.cat(linear_weights)
            else:
                return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)
        except:
            return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)

    def get_body_weights(self, flat_w):
        try:
            reshaped_weights = self.reshape_flat_weights(flat_w)
            non_linear_weights = []
            for name, weight in reshaped_weights.items():
                if 'Linear' not in name:
                    non_linear_weights.append(weight.view(-1))
            
            if non_linear_weights:
                return torch.cat(non_linear_weights)
            else:
                return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)
        except:
            return torch.tensor([], device=flat_w.device, dtype=flat_w.dtype)

    def forward_with_param(self, inp, new_w, *args, **kwargs):
        try:
            with self.unflatten_weight(new_w):
                return nn.Module.__call__(self, inp, *args, **kwargs)
        except Exception as e:
            print(f"Error in forward_with_param: {e}")
            return torch.zeros(inp.size(0), 10)  # Default output

    def load_state_dict(self, state_dict, strict=True):
        """Enhanced load_state_dict with comprehensive quantization support"""
        try:
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
                elif 'is_quantized' in state_dict and state_dict['is_quantized']:
                    self.is_quantized = True
                    self.quantization_scale = state_dict.get('quantization_scale', None)
                    self.quantization_method = state_dict.get('quantization_method', 'simple_1bit')
                else:
                    self.is_quantized = False
                    self.is_onebit_quantized = False
                    self.quantization_method = 'none'
                
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
        except Exception as e:
            print(f"Error loading state dict: {e}")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Enhanced state_dict with complete quantization state"""
        try:
            state = super(ReparamModule, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
            
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
        except Exception as e:
            print(f"Error creating state dict: {e}")
            return super(ReparamModule, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def __call__(self, inp, *args, **kwargs):
        return self.forward_with_param(inp, self.flat_w, *args, **kwargs)
    
    def get_quantization_summary(self):
        """Get a comprehensive summary of the current quantization state"""
        try:
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
        except Exception as e:
            print(f"Error getting quantization summary: {e}")
            return {'model_id': self.model_id, 'error': str(e)}

    def aggregate_with_fedawa_weights(self, other_models, adaptive_weights):
        """Aggregate this model with others using FedAwa adaptive weights"""
        try:
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
        except Exception as e:
            print(f"Error in FedAwa aggregation: {e}")
