import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from datasets import DatasetSplit
from utils import init_model
from utils import init_optimizer, model_parameter_vector
import torch.nn.functional as F
import time
import memory_profiler


def quantize(tensor, num_bits=8):
    """Uniform quantization of a tensor."""
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - tensor.min() / scale
    q_tensor = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    return q_tensor, scale, zero_point

def dequantize(q_tensor, scale, zero_point):
    """Dequantize a quantized tensor."""
    return scale * (q_tensor - zero_point)

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
            #Dequantize received weights
            for name, param in client_nodes[i].model.named_parameters():
                if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                    param.data = dequantize(param.data, param.scale, param.zero_point)

            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # Measure memory usage before quantization
            memory_before_quantization = memory_profiler.memory_usage()

            # Quantize weights and biases before sending to the server
            quantization_start_time = time.time()
            for name, param in client_nodes[i].model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    q_param, scale, zero_point = quantize(param.data)
                    param.data = q_param #Store quantized values
                    #Store scale and zero_point as attributes of the parameter
                    param.scale = scale
                    param.zero_point = zero_point
            quantization_end_time = time.time()

             # Measure memory usage after quantization
            memory_after_quantization = memory_profiler.memory_usage()

            print(f"Client {i}: Time taken for quantization: {quantization_end_time - quantization_start_time:.4f} seconds")
            print(f"Client {i}: Memory usage before quantization: {memory_before_quantization} MB")
            print(f"Client {i}: Memory usage after quantization: {memory_after_quantization} MB")
            

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in range(len(client_nodes)):
             #Dequantize received weights
            for name, param in client_nodes[i].model.named_parameters():
                if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
                    param.data = dequantize(param.data, param.scale, param.zero_point)

            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

             # Measure memory usage before quantization
            memory_before_quantization = memory_profiler.memory_usage()

            # Quantize weights and biases before sending to the server
            quantization_start_time = time.time()
            for name, param in client_nodes[i].model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    q_param, scale, zero_point = quantize(param.data)
                    param.data = q_param #Store quantized values
                     #Store scale and zero_point as attributes of the parameter
                    param.scale = scale
                    param.zero_point = zero_point
            quantization_end_time = time.time()

             # Measure memory usage after quantization
            memory_after_quantization = memory_profiler.memory_usage()

            print(f"Client {i}: Time taken for quantization: {quantization_end_time - quantization_start_time:.4f} seconds")
            print(f"Client {i}: Memory usage before quantization: {memory_before_quantization} MB")
            print(f"Client {i}: Memory usage after quantization: {memory_after_quantization} MB")


   
    else:
        raise ValueError('Undefined server method...')


    return client_nodes, train_loss

def Client_validate(args, client_nodes):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in range(len(client_nodes)):
        acc = validate(args, client_nodes[idx])
        # print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)
    # print("client personalization acc is ", client_acc)
    # exit()
    return avg_client_acc,client_acc




def DKL(_p, _q):
    # print(_p)
    # print(_q)
    # print(_p.log())
    # print(_q.log())
    # print(_p.log() - _q.log())
    return  torch.sum(_p * (_p.log() - _q.log()), dim=-1)


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
        # if "para" in args.server_method:
        #     output_local = node.model.forward_with_param(data.cuda(), node.model.get_param().cuda())
        # else:

        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)


# FedProx
def client_fedprox(global_model_param, args, node, loss = 0.0):
    node.model.train()
    # for name, param in node.model.named_parameters():
    #     if "layers.4" in name:
    #         param.requires_grad = False
    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        node.optimizer.step(global_model_param)

    return loss/len(train_loader)

