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

def calculate_model_size(model, consider_quantization=False):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
    for param in model.parameters():
        if consider_quantization and hasattr(param, 'scale'):
            # For quantized parameters: 1 bit per parameter + scale (4 bytes)
            total_size += (param.numel() / 8) + 4  # 1 bit per param + 4 bytes for scale
        else:
            if param.data.dtype == torch.float32:
                total_size += param.numel() * 4  # 4 bytes per float32
            elif param.data.dtype == torch.float16:
                total_size += param.numel() * 2  # 2 bytes per float16
            elif param.data.dtype == torch.int8:
                total_size += param.numel() * 1  # 1 byte per int8
            else:
                total_size += param.numel() * 4  # Default to 4 bytes
    return total_size / 1024 / 1024  # Convert to MB

def quantize(tensor):
    """1-bit quantization of a tensor."""
    scale = torch.mean(torch.abs(tensor))
    q_tensor = torch.sign(tensor)
    return q_tensor, scale

def dequantize(q_tensor, scale):
    """Dequantize a 1-bit quantized tensor."""
    return q_tensor * scale

def log_server_quantization_metrics(metrics_dict, operation="quantization"):
    """
    Log server quantization metrics in a structured format
    """
    print(f"\n📋 SERVER {operation.upper()} METRICS LOG:")
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
   
    print("=" * 80)

def export_server_metrics_to_csv(metrics_dict, operation, filename="server_quantization_metrics.csv"):
    """
    Export server metrics to CSV file for further analysis
    """
    import csv
    import os
   
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
   
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
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
            'model_size_change_percentage'
        ]
       
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
       
        # Prepare row data
        row_data = {'operation': operation}
        row_data.update(metrics_dict)
       
        writer.writerow(row_data)
   
    print(f"📊 Server metrics exported to {filename}")

def receive_client_models(args, client_nodes, select_list, size_weights):
    print(f"\n🔄 RECEIVING CLIENT MODELS:")
    print(f"Selected clients: {select_list}")
   
    client_params = []
    total_dequantization_time = 0
   
    for idx in select_list:
        print(f"\nProcessing client {idx}...")
       
        # Measure before dequantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        memory_before_dequant = get_memory_usage()
        tensor_memory_before_dequant = get_tensor_memory_usage()
        model_size_before_dequant = calculate_model_size(client_nodes[idx].model, consider_quantization=True)
       
        # Dequantize client model
        dequant_start_time = time.time()
        model_state = {}
        for name, param in client_nodes[idx].model.named_parameters():
            if hasattr(param, 'scale'):
                # Dequantize the parameter before storing it
                dequantized_param = dequantize(param.data, param.scale)
                model_state[name] = dequantized_param
            else:
                model_state[name] = param.data
        dequant_end_time = time.time()
       
        # Measure after dequantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        memory_after_dequant = get_memory_usage()
        tensor_memory_after_dequant = get_tensor_memory_usage()
        model_size_after_dequant = calculate_model_size(client_nodes[idx].model, consider_quantization=False)
       
        dequant_time = dequant_end_time - dequant_start_time
        total_dequantization_time += dequant_time
       
        # Calculate changes
        memory_change = memory_after_dequant - memory_before_dequant
        model_size_change = model_size_after_dequant - model_size_before_dequant
       
        print(f"Client {idx} dequantization:")
        print(f"  Time: {dequant_time:.4f}s")
        print(f"  Memory change: {memory_change:.2f} MB")
        print(f"  Model size change: {model_size_change:.2f} MB")
       
        client_params.append(model_state)

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]
   
    print(f"\nTotal client dequantization time: {total_dequantization_time:.4f}s")
    print(f"Aggregation weights: {agg_weights}")
   
    return agg_weights, client_params

def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    print(f"\n🔄 RECEIVING CLIENT MODELS (POOL):")
    print(f"Selected clients: {select_list}")
   
    client_params = []
    total_dequantization_time = 0
   
    for idx in select_list:
        print(f"\nProcessing client {idx}...")
       
        # Measure before dequantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        memory_before_dequant = get_memory_usage()
        dequant_start_time = time.time()
       
        model_state = {}
        for name, param in client_nodes[idx].model.named_parameters():
            if hasattr(param, 'scale'):
                # Dequantize the parameter
                dequantized_param = dequantize(param.data, param.scale)
                model_state[name] = dequantized_param
            else:
                model_state[name] = param.data
       
        dequant_end_time = time.time()
        dequant_time = dequant_end_time - dequant_start_time
        total_dequantization_time += dequant_time
       
        print(f"Client {idx} dequantization time: {dequant_time:.4f}s")
        client_params.append(model_state)

    agg_weights = [size_weights[idx] for idx in select_list]
   
    print(f"\nTotal client dequantization time: {total_dequantization_time:.4f}s")
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

def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=None, change=0):
    '''
    Enhanced server update functions with comprehensive quantization measurements
    '''
    global size_weights_global
    global global_T_weights
   
    print(f"\n🚀 STARTING SERVER UPDATE ROUND {rounds_num}")
    print(f"{'='*60}")
   
    server_round_start_time = time.time()
   
    if rounds_num==change:
        size_weights_global=size_weights
   
    # receive the local models from clients
    if args.server_method == 'fedawa':
        agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights_global)
    else:
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
    print(f"Aggregation weights: {agg_weights}")
   
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
    elif args.server_method == 'fedawa':
        if rounds_num==change:      
            global_T_weights=torch.tensor(agg_weights, dtype=torch.float32).to('cuda')
       
        avg_global_param, cur_global_T_weight = fedawa(args, client_params, agg_weights, central_node, rounds_num, global_T_weights)
        global_T_weights=cur_global_T_weight
        for i in range(len(select_list)):
            size_weights_global[select_list[i]] = global_T_weights[i]
        print("Global size weights:",size_weights_global)
        central_node.model.load_param(avg_global_param)
    else:
        raise ValueError('Undefined server method...')

    server_round_end_time = time.time()
    total_server_round_time = server_round_end_time - server_round_start_time
   
    print(f"\n🏁 SERVER ROUND {rounds_num} COMPLETED")
    print(f"Total server round time: {total_server_round_time:.4f} seconds")
    print(f"{'='*60}")
   
    return central_node

# FedAvg
def fedavg(parameters, list_nums_local_data):
    print(f"\n📊 PERFORMING FEDAVG AGGREGATION")
    aggregation_start_time = time.time()
   
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
   
    aggregation_end_time = time.time()
    print(f"FedAvg aggregation time: {aggregation_end_time - aggregation_start_time:.4f} seconds")
   
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
    print(f"\n🔀 PERFORMING FEDAWA AGGREGATION WITH COMPREHENSIVE MEASUREMENTS")
    print(f"{'='*80}")
   
    fedawa_start_time = time.time()
    param = central_node.model.get_param()

    # ============ BEFORE DEQUANTIZATION MEASUREMENTS ============
    print(f"\n📏 BEFORE DEQUANTIZATION MEASUREMENTS:")
   
    # Force garbage collection before measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory before dequantization
    memory_before_dequantization = get_memory_usage()
    tensor_memory_before_dequantization = get_tensor_memory_usage()
    model_size_before_dequantization = calculate_model_size(central_node.model, consider_quantization=True)

    print(f"Server: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
    print(f"Server: Tensor memory before dequantization: {tensor_memory_before_dequantization:.2f} MB")
    print(f"Server: Model size before dequantization: {model_size_before_dequantization:.2f} MB")

    # ============ DEQUANTIZATION PHASE ============
    print(f"\n⚡ DEQUANTIZATION PHASE:")
   
    # Dequantize global parameters (if needed)
    dequantization_start_time = time.time()
    # Placeholder for dequantization logic if central node is quantized
    # for name, param in central_node.model.named_parameters():
    #     if hasattr(param, 'scale'):
    #         param.data = dequantize(param.data, param.scale)
    dequantization_end_time = time.time()
    dequantization_time = dequantization_end_time - dequantization_start_time

    # ============ AFTER DEQUANTIZATION MEASUREMENTS ============
    print(f"\n📐 AFTER DEQUANTIZATION MEASUREMENTS:")
   
    # Force garbage collection after dequantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory after dequantization
    memory_after_dequantization = get_memory_usage()
    tensor_memory_after_dequantization = get_tensor_memory_usage()
    model_size_after_dequantization = calculate_model_size(central_node.model, consider_quantization=False)

    # Calculate dequantization impact
    memory_change_from_dequantization = memory_after_dequantization - memory_before_dequantization
    model_size_change_dequant = model_size_after_dequantization - model_size_before_dequantization
   
    # Log dequantization metrics
    dequant_metrics = {
        'memory_before_dequantization': memory_before_dequantization,
        'tensor_memory_before_dequantization': tensor_memory_before_dequantization,
        'model_size_before_dequantization': model_size_before_dequantization,
        'time_taken_for_dequantization': dequantization_time,
        'memory_after_dequantization': memory_after_dequantization,
        'tensor_memory_after_dequantization': tensor_memory_after_dequantization,
        'model_size_after_dequantization': model_size_after_dequantization,
        'memory_change_from_dequantization': memory_change_from_dequantization,
        'model_size_change': model_size_change_dequant
    }
   
    # Calculate percentages
    if memory_before_dequantization > 0:
        memory_change_percentage = (memory_change_from_dequantization / memory_before_dequantization) * 100
        dequant_metrics['memory_change_percentage'] = memory_change_percentage
   
    if model_size_before_dequantization > 0:
        model_size_change_percentage = (model_size_change_dequant / model_size_before_dequantization) * 100
        dequant_metrics['model_size_change_percentage'] = model_size_change_percentage
   
    log_server_quantization_metrics(dequant_metrics, "dequantization")
   
    try:
        export_server_metrics_to_csv(dequant_metrics, "dequantization")
    except Exception as e:
        print(f"Warning: Could not export dequantization metrics to CSV: {e}")

    print(f"Server: Time taken for dequantization: {dequantization_time:.4f} seconds")
    print(f"Server: Memory change from dequantization: {memory_change_from_dequantization:.2f} MB")
    print(f"Server: Model size change from dequantization: {model_size_change_dequant:.2f} MB")

    global_params = copy.deepcopy(param)

    # ============ FEDAWA COMPUTATION PHASE ============
    print(f"\n🧮 FEDAWA COMPUTATION PHASE:")
   
    computation_start_time = time.time()
   
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
        print(f"Server weight update iteration: {i}")
       
        probability_train = torch.nn.functional.softmax(T_weights, dim=0)
       
        C = _cost_matrix(global_flat_w.detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
       
        reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
        print(f"reg_loss: {reg_loss}")

        client_grad = local_param_list - global_flat_w
       
        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)  # weighted sum
       
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
       
        print(f"L2_distance: {l2_distance}")
        sim_loss = (torch.sum(probability_train*l2_distance, dim=(-2, -1)))

        print(f"Sim_loss: {sim_loss}")
     
        Loss = sim_loss + reg_loss
        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print(f"Step {i} Loss: {Loss}")

    global_T_weight = T_weights.data

    print(f"T_weights_after update: {global_T_weight}")
    print(f"probability_train_after update: {torch.nn.functional.softmax(global_T_weight, dim=0)}")

    # Aggregate parameters
    fedavg_global_params = copy.deepcopy(parameters[0])
    probability_train_final = torch.nn.functional.softmax(global_T_weight, dim=0)
   
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, weight in zip(parameters, probability_train_final):
            list_values_param.append(dict_local_params[name_param] * weight * args.gamma)
        value_global_param = sum(list_values_param) / sum(probability_train_final)
        fedavg_global_params[name_param] = value_global_param

    computation_end_time = time.time()
    computation_time = computation_end_time - computation_start_time
    print(f"FedAWA computation time: {computation_time:.4f} seconds")

    # ============ BEFORE QUANTIZATION MEASUREMENTS ============
    print(f"\n📏 BEFORE QUANTIZATION MEASUREMENTS:")
   
    # Force garbage collection before quantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory before quantization
    memory_before_quantization = get_memory_usage()
    tensor_memory_before_quantization = get_tensor_memory_usage()
    model_size_before_quantization = calculate_model_size(central_node.model, consider_quantization=False)

    print(f"Server: Memory usage before quantization: {memory_before_quantization:.2f} MB")
    print(f"Server: Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
    print(f"Server: Model size before quantization: {model_size_before_quantization:.2f} MB")

    # ============ QUANTIZATION PHASE ============
    print(f"\n⚡ QUANTIZATION PHASE:")
   
    # Quantize global parameters
    quantization_start_time = time.time()
    for name, param in central_node.model.named_parameters():
        if 'weight' in name or 'bias' in name:
            q_param, scale = quantize(param.data)
            param.data = q_param
            param.scale = scale
            param.requires_grad = False
    quantization_end_time = time.time()
    quantization_time = quantization_end_time - quantization_start_time

    # ============ AFTER QUANTIZATION MEASUREMENTS ============
    print(f"\n📐 AFTER QUANTIZATION MEASUREMENTS:")
   
    # Force garbage collection after quantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory after quantization
    memory_after_quantization = get_memory_usage()
    tensor_memory_after_quantization = get_tensor_memory_usage()
    model_size_after_quantization = calculate_model_size(central_node.model, consider_quantization=True)

    # Calculate quantization impact
    memory_reduction_from_quantization = memory_before_quantization - memory_after_quantization
    model_size_reduction = model_size_before_quantization - model_size_after_quantization
   
    # Log quantization metrics
    quant_metrics = {
        'memory_before_quantization': memory_before_quantization,
        'tensor_memory_before_quantization': tensor_memory_before_quantization,
        'model_size_before_quantization': model_size_before_quantization,
        'time_taken_for_quantization': quantization_time,
        'memory_after_quantization': memory_after_quantization,
        'tensor_memory_after_quantization': tensor_memory_after_quantization,
        'model_size_after_quantization': model_size_after_quantization,
        'memory_change_from_quantization': memory_reduction_from_quantization,
        'model_size_change': model_size_reduction
    }
   
    # Calculate percentages
    if memory_before_quantization > 0:
        memory_reduction_percentage = (memory_reduction_from_quantization / memory_before_quantization) * 100
        quant_metrics['memory_change_percentage'] = memory_reduction_percentage
   
    if model_size_before_quantization > 0:
        model_size_reduction_percentage = (model_size_reduction / model_size_before_quantization) * 100
        compression_ratio = (model_size_after_quantization / model_size_before_quantization) * 100
        quant_metrics['model_size_change_percentage'] = model_size_reduction_percentage
        quant_metrics['compression_ratio'] = compression_ratio
   
    log_server_quantization_metrics(quant_metrics, "quantization")
   
    try:
        export_server_metrics_to_csv(quant_metrics, "quantization")
    except Exception as e:
        print(f"Warning: Could not export quantization metrics to CSV: {e}")

    print(f"Global model: Time taken for quantization: {quantization_time:.4f} seconds")
    print(f"Server: Memory reduction from quantization: {memory_reduction_from_quantization:.2f} MB")
    print(f"Server: Model size reduction: {model_size_reduction:.2f} MB")

    # ============ FEDAWA SUMMARY ============
    fedawa_end_time = time.time()
    total_fedawa_time = fedawa_end_time - fedawa_start_time
   
    print(f"\n📈 FEDAWA SUMMARY:")
    print(f"{'─'*60}")
    print(f"REQUIRED SERVER MEASUREMENTS:")
    print(f"  • Memory usage before quantization: {memory_before_quantization:.2f} MB")
    print(f"  • Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
    print(f"  • Model size before quantization: {model_size_before_quantization:.2f} MB")
    print(f"  • Time taken for quantization: {quantization_time:.4f} seconds")
    print(f"  • Memory usage after quantization: {memory_after_quantization:.2f} MB")
    print(f"  • Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
    print(f"  • Model size after quantization: {model_size_after_quantization:.2f} MB")
    print(f"  • Memory reduction from quantization: {memory_reduction_from_quantization:.2f} MB")
    print(f"  • Model size reduction: {model_size_reduction:.2f} MB")
    print(f"")
    print(f"TIMING ANALYSIS:")
    print(f"  • Dequantization time: {dequantization_time:.4f} seconds")
    print(f"  • FedAWA computation time: {computation_time:.4f} seconds")
    print(f"  • Quantization time: {quantization_time:.4f} seconds")
    print(f"  • Total FedAWA time: {total_fedawa_time:.4f} seconds")
    print(f"{'='*80}")

    return fedavg_global_params, global_T_weight
