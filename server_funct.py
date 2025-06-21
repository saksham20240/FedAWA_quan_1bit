import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from torch.autograd import Variable

##############################################################################
# Server Update Functions
##############################################################################

def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=0):
    """Server update function with FedAwa support"""
    
    try:
        if args.server_method == 'fedawa':
            return fedawa_server_update(args, central_node, client_nodes, select_list, size_weights)
        elif args.server_method == 'fedavg':
            return fedavg_server_update(args, central_node, client_nodes, select_list, size_weights)
        else:
            # Default to FedAvg
            return fedavg_server_update(args, central_node, client_nodes, select_list, size_weights)
    except Exception as e:
        print(f"Error in Server_update: {e}")
        return central_node

def fedavg_server_update(args, central_node, client_nodes, select_list, size_weights):
    """Standard FedAvg server update"""
    
    try:
        # Get aggregation weights
        agg_weights = [size_weights[idx] for idx in select_list]
        agg_weights = [w/sum(agg_weights) for w in agg_weights]
        
        # Check if models are ReparamModule models
        is_reparam = hasattr(central_node.model, 'flat_w')
        
        if is_reparam:
            # ReparamModule aggregation
            central_node.model.flat_w.data.zero_()
            
            for i, client_idx in enumerate(select_list):
                if hasattr(client_nodes[client_idx].model, 'flat_w'):
                    central_node.model.flat_w.data += agg_weights[i] * client_nodes[client_idx].model.flat_w.data
        else:
            # Standard model aggregation
            global_param = {}
            
            # Initialize with first client's parameters - only tensors
            first_client_params = client_nodes[select_list[0]].model.state_dict()
            for name, param in first_client_params.items():
                if isinstance(param, torch.Tensor):
                    global_param[name] = torch.zeros_like(param)
            
            # Weighted aggregation - only for tensors
            for i, client_idx in enumerate(select_list):
                client_params = client_nodes[client_idx].model.state_dict()
                for name, param in client_params.items():
                    if name in global_param and isinstance(param, torch.Tensor):
                        global_param[name] += agg_weights[i] * param
            
            # Update central model - only load tensor parameters
            central_node.model.load_state_dict(global_param, strict=False)
        
        return central_node
    except Exception as e:
        print(f"Error in fedavg_server_update: {e}")
        return central_node

def fedawa_server_update(args, central_node, client_nodes, select_list, size_weights):
    """FedAwa server update with adaptive weighting"""
    
    try:
        from client_funct import compute_client_importance_weights
        
        # Get selected client nodes
        selected_clients = [client_nodes[idx] for idx in select_list]
        
        # Compute adaptive weights
        adaptive_weights, _, _, _ = compute_client_importance_weights(selected_clients, central_node)
        
        # Check if models are ReparamModule models
        is_reparam = hasattr(central_node.model, 'flat_w')
        
        if is_reparam:
            # Use ReparamModule-specific aggregation
            try:
                # Aggregate flat_w parameters
                central_node.model.flat_w.data.zero_()
                
                for i, client in enumerate(selected_clients):
                    if hasattr(client.model, 'flat_w'):
                        # Handle quantized models
                        if hasattr(client.model, 'is_onebit_quantized') and client.model.is_onebit_quantized:
                            # Dequantize for aggregation
                            if hasattr(client.model, 'g_vector') and hasattr(client.model, 'h_vector'):
                                # Simple dequantization for aggregation
                                effective_flat_w = client.model.flat_w.data * torch.mean(torch.abs(client.model.g_vector)) * torch.mean(torch.abs(client.model.h_vector))
                            else:
                                effective_flat_w = client.model.flat_w.data
                        elif hasattr(client.model, 'is_quantized') and client.model.is_quantized:
                            # Simple dequantization
                            if hasattr(client.model, 'quantization_scale'):
                                effective_flat_w = client.model.flat_w.data * client.model.quantization_scale
                            else:
                                effective_flat_w = client.model.flat_w.data
                        else:
                            effective_flat_w = client.model.flat_w.data
                        
                        central_node.model.flat_w.data += adaptive_weights[i] * effective_flat_w
                
            except Exception as e:
                print(f"Warning: Error in ReparamModule aggregation: {e}, using standard aggregation")
                return fedavg_server_update(args, central_node, client_nodes, select_list, size_weights)
        else:
            # Standard parameter aggregation with adaptive weights
            global_param = {}
            
            # Initialize with first client's parameters
            first_client_params = selected_clients[0].model.state_dict()
            for name in first_client_params:
                if isinstance(first_client_params[name], torch.Tensor):
                    global_param[name] = torch.zeros_like(first_client_params[name])
            
            # Weighted aggregation with adaptive weights
            for i, client in enumerate(selected_clients):
                client_params = client.model.state_dict()
                for name in client_params:
                    if name in global_param and isinstance(client_params[name], torch.Tensor):
                        global_param[name] += adaptive_weights[i] * client_params[name]
            
            # Update central model
            central_node.model.load_state_dict(global_param, strict=False)
        
        return central_node
    except Exception as e:
        print(f"Error in fedawa_server_update: {e}")
        return central_node

##############################################################################
# Client Update Functions
##############################################################################

def Client_update(args, client_nodes, central_node):
    """Client update function"""
    
    try:
        total_loss = 0.0
        num_clients = len(client_nodes)
        
        # Distribute central model to all clients
        if hasattr(central_node.model, 'flat_w'):
            # ReparamModule - use flat_w distribution
            central_flat_w = central_node.model.flat_w.data.clone()
            for client_id in range(num_clients):
                client_nodes[client_id].model.flat_w.data.copy_(central_flat_w)
                
                # Copy quantization state safely
                try:
                    if hasattr(central_node.model, 'is_onebit_quantized'):
                        client_nodes[client_id].model.is_onebit_quantized = central_node.model.is_onebit_quantized
                    if hasattr(central_node.model, 'is_quantized'):
                        client_nodes[client_id].model.is_quantized = central_node.model.is_quantized
                    if hasattr(central_node.model, 'quantization_method'):
                        client_nodes[client_id].model.quantization_method = central_node.model.quantization_method
                    
                    if hasattr(central_node.model, 'g_vector') and central_node.model.g_vector is not None:
                        if not hasattr(client_nodes[client_id].model, 'g_vector') or client_nodes[client_id].model.g_vector is None:
                            client_nodes[client_id].model.g_vector = nn.Parameter(central_node.model.g_vector.data.clone())
                        else:
                            client_nodes[client_id].model.g_vector.data.copy_(central_node.model.g_vector.data)
                    
                    if hasattr(central_node.model, 'h_vector') and central_node.model.h_vector is not None:
                        if not hasattr(client_nodes[client_id].model, 'h_vector') or client_nodes[client_id].model.h_vector is None:
                            client_nodes[client_id].model.h_vector = nn.Parameter(central_node.model.h_vector.data.clone())
                        else:
                            client_nodes[client_id].model.h_vector.data.copy_(central_node.model.h_vector.data)
                    
                    if hasattr(central_node.model, 'quantization_scale') and central_node.model.quantization_scale is not None:
                        client_nodes[client_id].model.quantization_scale = central_node.model.quantization_scale
                        
                except Exception as copy_error:
                    print(f"Warning: Error copying quantization state to client {client_id}: {copy_error}")
        else:
            # Standard model - use state dict
            try:
                central_state = central_node.model.state_dict()
                # Filter out non-tensor items for standard models
                tensor_state = {k: v for k, v in central_state.items() if isinstance(v, torch.Tensor)}
                
                for client_id in range(num_clients):
                    client_nodes[client_id].model.load_state_dict(tensor_state, strict=False)
            except Exception as e:
                print(f"Warning: Error distributing standard model: {e}")
        
        # Local training
        for client_id in range(num_clients):
            try:
                if args.use_onebit_training:
                    from client_funct import client_localTrain_onebit
                    loss = client_localTrain_onebit(args, client_nodes[client_id])
                else:
                    loss = local_train_standard(args, client_nodes[client_id])
                
                total_loss += loss
            except Exception as train_error:
                print(f"Warning: Error training client {client_id}: {train_error}")
                total_loss += 2.0  # Default loss value
        
        avg_loss = total_loss / num_clients
        return client_nodes, avg_loss
    except Exception as e:
        print(f"Error in Client_update: {e}")
        return client_nodes, 0.0

def local_train_standard(args, node):
    """Standard local training function"""
    
    try:
        node.model.train()
        total_loss = 0.0
        
        for epoch in range(getattr(args, 'E', 5)):
            for idx, (data, target) in enumerate(node.local_data):
                node.optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                output = node.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                total_loss += loss.item()
                
                node.optimizer.step()
        
        return total_loss / (getattr(args, 'E', 5) * len(node.local_data))
    except Exception as e:
        print(f"Error in local_train_standard: {e}")
        return 0.0

##############################################################################
# Validation Functions
##############################################################################

def Client_validate(args, client_nodes):
    """Client validation function"""
    
    try:
        total_acc = 0.0
        client_accs = []
        
        for client_id in range(len(client_nodes)):
            if args.use_onebit_training:
                from client_funct import validate_onebit
                acc = validate_onebit(args, client_nodes[client_id])
            else:
                acc = validate_standard(args, client_nodes[client_id])
            
            client_accs.append(acc)
            total_acc += acc
        
        avg_acc = total_acc / len(client_nodes)
        return avg_acc, client_accs
    except Exception as e:
        print(f"Error in Client_validate: {e}")
        return 0.0, [0.0] * len(client_nodes)

def validate_standard(args, node):
    """Standard validation function"""
    
    try:
        node.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in node.local_data:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                output = node.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy
    except Exception as e:
        print(f"Error in validate_standard: {e}")
        return 0.0

##############################################################################
# Utility Functions
##############################################################################

def to_var(x, requires_grad=True):
    """Convert to Variable with GPU support"""
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
    """Compute cost matrix for optimal transport"""
    try:
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        return C
    except Exception as e:
        print(f"Error in _cost_matrix: {e}")
        return torch.zeros(x.size(0), y.size(0))
