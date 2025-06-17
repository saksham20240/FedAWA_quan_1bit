import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from datasets import DatasetSplit
from utils import init_model
from utils import init_optimizer, model_parameter_vector
import torch.nn.functional as F
import time
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_tensor_memory_usage():
    """Get GPU memory usage if CUDA is available, otherwise return 0"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0

def calculate_model_size(model):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
    for param in model.parameters():
        if param.data.dtype == torch.float32:
            total_size += param.numel() * 4  # 4 bytes per float32
        elif param.data.dtype == torch.float16:
            total_size += param.numel() * 2  # 2 bytes per float16
        elif param.data.dtype == torch.int8:
            total_size += param.numel() * 1  # 1 byte per int8
        else:
            total_size += param.numel() * 4  # Default to 4 bytes
    return total_size / 1024 / 1024  # Convert to MB

def quantize(tensor, num_bits=8):
    """Uniform quantization of a tensor."""
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - tensor.min() / scale
    q_tensor = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    return q_tensor.to(torch.int8), scale, zero_point  # Convert to int8

def dequantize(q_tensor, scale, zero_point):
    """Dequantize a quantized tensor."""
    return scale * (q_tensor.float() - zero_point)

def receive_server_model(args, client_nodes, central_node):
    for idx in range(len(client_nodes)):
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_nodes[idx].model.load_param(copy.deepcopy(central_node.model.get_param(clone = True)))
        else:
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node):
    '''
    client update functions
    '''
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    if args.client_method == 'local_train':
        client_losses = []
        for i in range(len(client_nodes)):
            # Dequantize received weights
            for name, param in client_nodes[i].model.named_parameters():
                if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                    param.data = dequantize(param.data, param.scale, param.zero_point)

            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # Force garbage collection before measurement
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory before quantization
            memory_before_quantization = get_memory_usage()
            tensor_memory_before_quant = get_tensor_memory_usage()
            model_size_before_quant = calculate_model_size(client_nodes[i].model)

            print(f"Client {i}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
            print(f"Client {i}: Tensor memory before quantization: {tensor_memory_before_quant:.2f} MB")
            print(f"Client {i}: Model size before quantization: {model_size_before_quant:.2f} MB")

            # Quantize weights and biases before sending to the server
            quantization_start_time = time.time()
            for name, param in client_nodes[i].model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    q_param, scale, zero_point = quantize(param.data)
                    param.data = q_param  # Store quantized values
                    # Store scale and zero_point as attributes of the parameter
                    param.scale = scale
                    param.zero_point = zero_point
                    # Ensure quantized parameters do not track gradients
                    param.requires_grad = False
            quantization_end_time = time.time()

            # Force garbage collection after quantization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory after quantization
            memory_after_quantization = get_memory_usage()
            tensor_memory_after_quant = get_tensor_memory_usage()
            model_size_after_quant = calculate_model_size(client_nodes[i].model)

            print(f"Client {i}: Time taken for quantization: {quantization_end_time - quantization_start_time:.4f} seconds")
            print(f"Client {i}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
            print(f"Client {i}: Tensor memory after quantization: {tensor_memory_after_quant:.2f} MB")
            print(f"Client {i}: Model size after quantization: {model_size_after_quant:.2f} MB")
            print(f"Client {i}: Memory reduction from quantization: {memory_before_quantization - memory_after_quantization:.2f} MB")
            print(f"Client {i}: Model size reduction: {model_size_before_quant - model_size_after_quant:.2f} MB")

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in range(len(client_nodes)):
            # Dequantize received weights
            for name, param in client_nodes[i].model.named_parameters():
                if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                    param.data = dequantize(param.data, param.scale, param.zero_point)

            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # Force garbage collection before measurement
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory before quantization
            memory_before_quantization = get_memory_usage()
            tensor_memory_before_quant = get_tensor_memory_usage()
            model_size_before_quant = calculate_model_size(client_nodes[i].model)

            print(f"Client {i}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
            print(f"Client {i}: Tensor memory before quantization: {tensor_memory_before_quant:.2f} MB")
            print(f"Client {i}: Model size before quantization: {model_size_before_quant:.2f} MB")

            # Quantize weights and biases before sending to the server
            quantization_start_time = time.time()
            for name, param in client_nodes[i].model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    q_param, scale, zero_point = quantize(param.data)
                    param.data = q_param  # Store quantized values
                    # Store scale and zero_point as attributes of the parameter
                    param.scale = scale
                    param.zero_point = zero_point
                    # Ensure quantized parameters do not track gradients
                    param.requires_grad = False
            quantization_end_time = time.time()

            # Force garbage collection after quantization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory after quantization
            memory_after_quantization = get_memory_usage()
            tensor_memory_after_quant = get_tensor_memory_usage()
            model_size_after_quant = calculate_model_size(client_nodes[i].model)

            print(f"Client {i}: Time taken for quantization: {quantization_end_time - quantization_start_time:.4f} seconds")
            print(f"Client {i}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
            print(f"Client {i}: Tensor memory after quantization: {tensor_memory_after_quant:.2f} MB")
            print(f"Client {i}: Model size after quantization: {model_size_after_quant:.2f} MB")
            print(f"Client {i}: Memory reduction from quantization: {memory_before_quantization - memory_after_quantization:.2f} MB")
            print(f"Client {i}: Model size reduction: {model_size_before_quant - model_size_after_quant:.2f} MB")

    else:
        raise ValueError('Undefined client method...')

    return client_nodes, train_loss

def Client_validate(args, client_nodes):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in range(len(client_nodes)):
        acc = validate(args, client_nodes[idx])
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)
    return avg_client_acc, client_acc

def DKL(_p, _q):
    return torch.sum(_p * (_p.log() - _q.log()), dim=-1)

# Vanilla local training
def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()

        output_local = node.model(data)

        loss_local = F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

# FedProx
def client_fedprox(global_model_param, args, node, loss = 0.0):
    node.model.train()
    
    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local = F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        node.optimizer.step(global_model_param)

    return loss/len(train_loader)
