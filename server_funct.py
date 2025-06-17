import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.autograd import Variable
from utils import validate
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

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        model_state = {}
        for name, param in client_nodes[idx].model.named_parameters():
            if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                # Dequantize the parameter before storing it
                dequantized_param = dequantize(param.data, param.scale, param.zero_point)
                model_state[name] = dequantized_param
            else:
                model_state[name] = param.data

        client_params.append(model_state)

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params

def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        model_state = {}
        for name, param in client_nodes[idx].model.named_parameters():
            if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                # Dequantize the parameter
                dequantized_param = dequantize(param.data, param.scale, param.zero_point)
                model_state[name] = dequantized_param
            else:
                model_state[name] = param.data

        client_params.append(model_state)

    agg_weights = [size_weights[idx] for idx in select_list]

    return agg_weights, client_params
    
def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(param.data))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params

global_T_weights_dict={}

def Server_update(args, central_node, client_nodes, select_list, size_weights,rounds_num=None,change=0):
    '''
    server update functions for baselines
    '''
    global size_weights_global
    global global_T_weights
    if rounds_num==change:
        size_weights_global=size_weights
    
    # receive the local models from clients
    if args.server_method == 'fedawa':
        agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights_global)
    else:
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
    print(agg_weights)
    
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
    elif args.server_method == 'fedawa':
        if rounds_num==change:       
            global_T_weights=torch.tensor(agg_weights, dtype=torch.float32).to('cuda')
        
        avg_global_param,cur_global_T_weight = fedawa(args,client_params, agg_weights,central_node,rounds_num,global_T_weights)
        global_T_weights=cur_global_T_weight
        for i in range(len(select_list)):
            size_weights_global[select_list[i]] = global_T_weights[i]
        print("Global size weights:",size_weights_global)
        central_node.model.load_param(avg_global_param)
    else:
        raise ValueError('Undefined server method...')

    return central_node

# FedAvg
def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

def unflatten_weight(M, flat_w):
    ws = (t.view(s) for (t, s) in zip(flat_w.split(M._weights_numels), M._weights_shapes))
    
    for (m, n), w in zip(M._weights_module_names, ws):
        if 'Batch' in str(type(m)):
            print(m,n,w)
        setattr(m, n, w)

def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
    d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    if dis == 'cos':
        C = 1-d_cosine(x_col, y_lin)
    elif dis == 'euc':
        C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
    return C

def fedawa(args, parameters, list_nums_local_data, central_node, rounds, global_T_weight):
    param = central_node.model.get_param()

    # Force garbage collection before measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory before dequantization
    memory_before_dequantization = get_memory_usage()
    tensor_memory_before_dequant = get_tensor_memory_usage()
    model_size_before_dequant = calculate_model_size(central_node.model)

    print(f"Server: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
    print(f"Server: Tensor memory before dequantization: {tensor_memory_before_dequant:.2f} MB")
    print(f"Server: Model size before dequantization: {model_size_before_dequant:.2f} MB")

    # Dequantize global parameters
    dequantization_start_time = time.time()
    for name, param in central_node.model.named_parameters():
        if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
            param.data = dequantize(param.data, param.scale, param.zero_point)
    dequantization_end_time = time.time()

    # Force garbage collection after dequantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory after dequantization
    memory_after_dequantization = get_memory_usage()
    tensor_memory_after_dequant = get_tensor_memory_usage()
    model_size_after_dequant = calculate_model_size(central_node.model)

    print(f"Server: Time taken for dequantization: {dequantization_end_time - dequantization_start_time:.4f} seconds")
    print(f"Server: Memory usage after dequantization: {memory_after_dequantization:.2f} MB")
    print(f"Server: Tensor memory after dequantization: {tensor_memory_after_dequant:.2f} MB")
    print(f"Server: Model size after dequantization: {model_size_after_dequant:.2f} MB")
    print(f"Server: Memory increase from dequantization: {memory_after_dequantization - memory_before_dequantization:.2f} MB")

    global_params = copy.deepcopy(param)

    # Force garbage collection before quantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory before quantization
    memory_before_quantization = get_memory_usage()
    tensor_memory_before_quant = get_tensor_memory_usage()
    model_size_before_quant = calculate_model_size(central_node.model)

    print(f"Server: Memory usage before quantization: {memory_before_quantization:.2f} MB")
    print(f"Server: Tensor memory before quantization: {tensor_memory_before_quant:.2f} MB")
    print(f"Server: Model size before quantization: {model_size_before_quant:.2f} MB")

    # Quantize global parameters
    quantization_start_time = time.time()
    for name, param in central_node.model.named_parameters():
        if 'weight' in name or 'bias' in name:
            q_param, scale, zero_point = quantize(param.data)
            param.data = q_param
            param.scale = scale
            param.zero_point = zero_point
    quantization_end_time = time.time()

    # Force garbage collection after quantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory after quantization
    memory_after_quantization = get_memory_usage()
    tensor_memory_after_quant = get_tensor_memory_usage()
    model_size_after_quant = calculate_model_size(central_node.model)

    print(f"Global model: Time taken for quantization: {quantization_end_time - quantization_start_time:.4f} seconds")
    print(f"Server: Memory usage after quantization: {memory_after_quantization:.2f} MB")
    print(f"Server: Tensor memory after quantization: {tensor_memory_after_quant:.2f} MB")
    print(f"Server: Model size after quantization: {model_size_after_quant:.2f} MB")
    print(f"Server: Memory reduction from quantization: {memory_before_quantization - memory_after_quantization:.2f} MB")
    print(f"Server: Model size reduction: {model_size_before_quant - model_size_after_quant:.2f} MB")

    # Rest of fedawa implementation
    flat_w_list = []
    
    # Flatten client parameters
    for dict_local_params in parameters:
        flat_w = torch.cat([w.flatten() for w in dict_local_params.values()])
        flat_w_list.append(flat_w)

    local_param_list = torch.stack(flat_w_list)
    
    T_weights = to_var(global_T_weight)
    
    if args.server_optimizer=='sgd':
        Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif args.server_optimizer=='adam':
        Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))
    
    print("T_weights_before update:",torch.nn.functional.softmax(T_weights, dim=0))

    # Flatten global parameters
    global_flat_w = torch.cat([w.flatten() for w in global_params.values()])
    
    # Server update iterations
    for i in range(args.server_epochs):
        print("server weight update:",i)
        
        probability_train = torch.nn.functional.softmax(T_weights, dim=0)
        
        C = _cost_matrix(global_flat_w.detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
        
        reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
        print("reg_loss:",reg_loss)

        client_grad = local_param_list - global_flat_w
        
        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)  # weighted sum
        
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
        
        print("L2_distance:",l2_distance)
        sim_loss = (torch.sum(probability_train*l2_distance, dim=(-2, -1)))

        print("Sim_loss:",sim_loss)
     
        Loss = sim_loss + reg_loss
        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print("step " + str(i) + " Loss:" + str(Loss))

    global_T_weight = T_weights.data

    print("T_weights_after update:", global_T_weight)
    print("probability_train_after update:", torch.nn.functional.softmax(global_T_weight, dim=0))

    # Aggregate parameters
    fedavg_global_params = copy.deepcopy(parameters[0])
    probability_train_final = torch.nn.functional.softmax(global_T_weight, dim=0)
    
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, weight in zip(parameters, probability_train_final):
            list_values_param.append(dict_local_params[name_param] * weight * args.gamma)
        value_global_param = sum(list_values_param) / sum(probability_train_final)
        fedavg_global_params[name_param] = value_global_param

    return fedavg_global_params, global_T_weight
