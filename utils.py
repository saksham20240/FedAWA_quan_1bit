import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import psutil
import gc
from torch.backends import cudnn
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn
from models_dict.vit import ViT, ViT_fedlaw

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

def calculate_model_size(model, consider_quantization=False):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
    for param in model.parameters():
        if consider_quantization and (hasattr(param, 'scale') or hasattr(param, 'is_onebit')):
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
                total_size += param.numel() * 4  # 4 bytes per float32
            elif param.data.dtype == torch.float16:
                total_size += param.numel() * 2  # 2 bytes per float16
            elif param.data.dtype == torch.int8:
                total_size += param.numel() * 1  # 1 byte per int8
            else:
                total_size += param.numel() * 4  # Default to 4 bytes
    return total_size / 1024 / 1024  # Convert to MB

def log_dequantization_metrics(node_id, metrics_dict, operation_context="validation"):
    """
    Log dequantization metrics in a structured format
    """
    print(f"\n📋 DEQUANTIZATION METRICS LOG FOR {operation_context.upper()} - NODE {node_id}:")
    print("=" * 80)
   
    # Core measurements
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
   
    # Additional analysis
    if 'memory_change_percentage' in metrics_dict:
        print(f"\nADDITIONAL ANALYSIS:")
        print(f"  Memory change percentage: {metrics_dict['memory_change_percentage']:.2f}%")
        print(f"  Model size change percentage: {metrics_dict['model_size_change_percentage']:.2f}%")
   
    print("=" * 80)

def export_dequantization_metrics_to_csv(node_id, metrics_dict, operation_context, filename="dequantization_metrics.csv"):
    """
    Export dequantization metrics to CSV file for further analysis
    """
    import csv
    import os
   
    # Check if file exists to determine if we need headers
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
       
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
       
        # Prepare row data
        row_data = {
            'node_id': node_id,
            'operation_context': operation_context
        }
        row_data.update(metrics_dict)
       
        writer.writerow(row_data)
   
    print(f"📊 Dequantization metrics exported to {filename}")

def measure_dequantization_comprehensive(model, node_id, operation_context="validation"):
    """
    Comprehensive measurement wrapper for dequantization operations
    """
    # ============ BEFORE DEQUANTIZATION MEASUREMENTS ============
    print(f"\n📏 BEFORE DEQUANTIZATION MEASUREMENTS ({operation_context.upper()}):")
   
    # Force garbage collection before measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
   
    # Measure memory before dequantization
    memory_before_dequantization = get_memory_usage()
    tensor_memory_before_dequantization = get_tensor_memory_usage()
    model_size_before_dequantization = calculate_model_size(model, consider_quantization=True)
   
    print(f"Node {node_id}: Memory usage before dequantization: {memory_before_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory before dequantization: {tensor_memory_before_dequantization:.2f} MB")
    print(f"Node {node_id}: Model size before dequantization: {model_size_before_dequantization:.2f} MB")
   
    # ============ DEQUANTIZATION PHASE ============
    print(f"\n⚡ DEQUANTIZATION PHASE ({operation_context.upper()}):")
   
    dequantization_start_time = time.time()
   
    # Perform dequantization
    for name, param in model.named_parameters():
        if hasattr(param, 'scale') and hasattr(param, 'zero_point'):
            param.data = dequantize(param.data, param.scale, param.zero_point)
        elif hasattr(param, 'scale') and not hasattr(param, 'zero_point'):
            # Simple 1-bit dequantization
            param.data = dequantize_simple(param.data, param.scale)
        elif hasattr(param, 'is_onebit') and param.is_onebit:
            # OneBit dequantization
            param.data = onebit_dequantize(param.data, param.g_vector, param.h_vector)
   
    dequantization_end_time = time.time()
    dequantization_time = dequantization_end_time - dequantization_start_time
   
    print(f"Node {node_id}: Time taken for dequantization: {dequantization_time:.4f} seconds")
   
    # ============ AFTER DEQUANTIZATION MEASUREMENTS ============
    print(f"\n📐 AFTER DEQUANTIZATION MEASUREMENTS ({operation_context.upper()}):")
   
    # Force garbage collection after dequantization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
   
    # Measure memory after dequantization
    memory_after_dequantization = get_memory_usage()
    tensor_memory_after_dequantization = get_tensor_memory_usage()
    model_size_after_dequantization = calculate_model_size(model, consider_quantization=False)
   
    print(f"Node {node_id}: Memory usage after dequantization: {memory_after_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory after dequantization: {tensor_memory_after_dequantization:.2f} MB")
    print(f"Node {node_id}: Model size after dequantization: {model_size_after_dequantization:.2f} MB")
   
    # ============ DEQUANTIZATION IMPACT ANALYSIS ============
    print(f"\n📊 DEQUANTIZATION IMPACT ANALYSIS ({operation_context.upper()}):")
   
    # Calculate changes
    memory_change_from_dequantization = memory_after_dequantization - memory_before_dequantization
    tensor_memory_change = tensor_memory_after_dequantization - tensor_memory_before_dequantization
    model_size_change = model_size_after_dequantization - model_size_before_dequantization
   
    print(f"Node {node_id}: Memory change from dequantization: {memory_change_from_dequantization:.2f} MB")
    print(f"Node {node_id}: Tensor memory change: {tensor_memory_change:.2f} MB")
    print(f"Node {node_id}: Model size change: {model_size_change:.2f} MB")
   
    # Calculate percentage changes
    memory_change_percentage = 0
    model_size_change_percentage = 0
   
    if memory_before_dequantization > 0:
        memory_change_percentage = (memory_change_from_dequantization / memory_before_dequantization) * 100
        print(f"Node {node_id}: Memory change percentage: {memory_change_percentage:.2f}%")
   
    if model_size_before_dequantization > 0:
        model_size_change_percentage = (model_size_change / model_size_before_dequantization) * 100
        print(f"Node {node_id}: Model size change percentage: {model_size_change_percentage:.2f}%")
   
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
        'model_size_change_percentage': model_size_change_percentage
    }
   
    # ============ STRUCTURED LOGGING ============
    log_dequantization_metrics(node_id, metrics_dict, operation_context)
   
    # Export to CSV (optional - can be enabled/disabled)
    try:
        export_dequantization_metrics_to_csv(node_id, metrics_dict, operation_context)
    except Exception as e:
        print(f"Warning: Could not export dequantization metrics to CSV: {e}")
   
    return dequantization_time

##############################################################################
# Enhanced Quantization functions
##############################################################################

def dequantize(q_tensor, scale, zero_point=0):
    """Dequantize a quantized tensor with optional zero_point."""
    if zero_point != 0:
        return scale * (q_tensor.float() - zero_point)
    else:
        return q_tensor * scale

def dequantize_simple(q_tensor, scale):
    """Simple 1-bit dequantization."""
    return q_tensor * scale

def onebit_dequantize(sign_matrix, g_vector, h_vector):
    """
    OneBit dequantization (approximation for validation)
    """
    if len(sign_matrix.shape) == 2:
        outer_product = torch.outer(h_vector, g_vector)
        reconstructed = sign_matrix * outer_product
    else:
        # For other shapes, use element-wise scaling
        scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
        reconstructed = sign_matrix * scale
   
    return reconstructed

##############################################################################
# Tools
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

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
# Initialization function
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
            model = resnet.WRN56_4(num_classes)  # Fixed: was ResNet56_4
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
# Training function
##############################################################################

def generate_selectlist(client_node, ratio=0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
    return select_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99  # 0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    # print('Learning rate={:.4f}'.format(args.lr))

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
# Enhanced Validation functions with comprehensive measurements
##############################################################################

def validate(args, node, which_dataset='validate'):
    print(f"\n🔍 STARTING VALIDATION WITH COMPREHENSIVE MEASUREMENTS")
    print(f"{'='*70}")
   
    validation_start_time = time.time()
    node.model.cuda().eval()

    # Get node identifier
    node_id = getattr(node, 'id', 'unknown')
   
    # Perform comprehensive dequantization measurements
    dequantization_time = measure_dequantization_comprehensive(
        node.model,
        node_id,
        operation_context=f"validation_{which_dataset}"
    )

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
       
        acc = correct / len(test_loader.dataset) * 100
   
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
   
    validation_end_time = time.time()
    total_validation_time = validation_end_time - validation_start_time
   
    # ============ VALIDATION SUMMARY ============
    print(f"\n📈 VALIDATION SUMMARY FOR NODE {node_id}:")
    print(f"{'─'*70}")
    print(f"REQUIRED DEQUANTIZATION MEASUREMENTS:")
    print(f"  [All measurements logged above in comprehensive format]")
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
    print(f"  • Dequantization overhead: {(dequantization_time/total_validation_time)*100:.1f}%")
    print(f"{'='*70}")
   
    return acc

def testloss(args, node, which_dataset='validate'):
    print(f"\n🔍 STARTING TEST LOSS COMPUTATION WITH COMPREHENSIVE MEASUREMENTS")
    print(f"{'='*70}")
   
    testloss_start_time = time.time()
    node.model.cuda().eval()
   
    # Get node identifier
    node_id = getattr(node, 'id', 'unknown')
   
    # Perform comprehensive dequantization measurements
    dequantization_time = measure_dequantization_comprehensive(
        node.model,
        node_id,
        operation_context=f"testloss_{which_dataset}"
    )
   
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
    print(f"\n📈 TEST LOSS SUMMARY FOR NODE {node_id}:")
    print(f"{'─'*70}")
    print(f"REQUIRED DEQUANTIZATION MEASUREMENTS:")
    print(f"  [All measurements logged above in comprehensive format]")
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
    print(f"  • Dequantization overhead: {(dequantization_time/total_testloss_time)*100:.1f}%")
    print(f"{'='*70}")
   
    return loss_value

# Functions for FedLAW with param as an input
def validate_with_param(args, node, param, which_dataset='validate'):
    print(f"\n🔍 STARTING VALIDATION WITH PARAM (FedLAW)")
    print(f"{'='*60}")
   
    validation_start_time = time.time()
    node.model.cuda().eval()
   
    # Get node identifier
    node_id = getattr(node, 'id', 'unknown')
   
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    # Measure validation performance with parameters
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
       
        acc = correct / len(test_loader.dataset) * 100
   
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
    print(f"\n🔍 STARTING TEST LOSS WITH PARAM (FedLAW)")
    print(f"{'='*60}")
   
    testloss_start_time = time.time()
    node.model.cuda().eval()
   
    # Get node identifier  
    node_id = getattr(node, 'id', 'unknown')
   
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    # Measure test loss computation with parameters
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
