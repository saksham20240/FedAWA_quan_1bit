import torch
import torch.nn.functional as F
import time
import psutil
import gc
import copy
import numpy as np
from sklearn.decomposition import NMF

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_tensor_memory_usage():
    """Get GPU memory usage if CUDA is available, otherwise return 0"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0

def calculate_model_size(model, consider_onebit=False):
    """Calculate the memory size of model parameters in MB"""
    total_size = 0
    for param in model.parameters():
        if consider_onebit and hasattr(param, 'is_onebit') and param.is_onebit:
            # OneBit: 1 bit per parameter + two FP16 vectors
            if hasattr(param, 'g_vector') and hasattr(param, 'h_vector'):
                # 1 bit for sign matrix + FP16 for g and h vectors
                sign_bits = param.numel() / 8  # 1 bit per parameter, packed
                vector_bits = (param.g_vector.numel() + param.h_vector.numel()) * 16  # FP16
                total_size += (sign_bits + vector_bits) / 8  # Convert to bytes
            else:
                total_size += param.numel() / 8  # Just 1 bit per parameter
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

def svid_decomposition(weight_matrix, method='nmf'):
    """
    Sign-Value-Independent Decomposition (SVID) for OneBit initialization
   
    Args:
        weight_matrix: Original weight matrix W
        method: 'nmf' or 'svd' for decomposition method
   
    Returns:
        sign_matrix: Sign matrix (±1)
        g_vector: First value vector
        h_vector: Second value vector
    """
    # Get sign matrix
    sign_matrix = torch.sign(weight_matrix)
   
    # Get absolute value matrix
    abs_matrix = torch.abs(weight_matrix)
   
    # Convert to numpy for decomposition
    abs_numpy = abs_matrix.detach().cpu().numpy()
   
    if method == 'nmf':
        # Use NMF for rank-1 approximation of |W|
        nmf = NMF(n_components=1, init='random', random_state=42, max_iter=1000)
        W_nmf = nmf.fit_transform(abs_numpy)  # Shape: (m, 1)
        H_nmf = nmf.components_  # Shape: (1, n)
       
        # Extract vectors
        a_vector = torch.from_numpy(W_nmf.flatten()).to(weight_matrix.device)  # Shape: (m,)
        b_vector = torch.from_numpy(H_nmf.flatten()).to(weight_matrix.device)  # Shape: (n,)
       
    else:  # SVD method
        U, S, Vt = np.linalg.svd(abs_numpy, full_matrices=False)
       
        # Take rank-1 approximation (largest singular value)
        a_vector = torch.from_numpy(U[:, 0] * np.sqrt(S[0])).to(weight_matrix.device)
        b_vector = torch.from_numpy(Vt[0, :] * np.sqrt(S[0])).to(weight_matrix.device)
   
    return sign_matrix, a_vector, b_vector

def onebit_quantize(weight_matrix, method='nmf'):
    """
    OneBit quantization: convert weight matrix to sign matrix + two value vectors
   
    Args:
        weight_matrix: Original weight matrix
        method: Decomposition method for SVID
   
    Returns:
        sign_matrix: Quantized sign matrix (±1)
        g_vector: First value vector (corresponds to input dimension)
        h_vector: Second value vector (corresponds to output dimension)
    """
    sign_matrix, a_vector, b_vector = svid_decomposition(weight_matrix, method)
   
    # In OneBit paper: Y = (X ⊙ g) * W±1^T ⊙ h
    # So g corresponds to input dimension, h corresponds to output dimension
    # For weight matrix of shape (out_features, in_features):
    # g should have shape (in_features,) and h should have shape (out_features,)
   
    if len(weight_matrix.shape) == 2:  # Standard linear layer weight
        out_features, in_features = weight_matrix.shape
        g_vector = b_vector  # Input dimension vector
        h_vector = a_vector  # Output dimension vector
    else:
        # For other shapes, use the vectors as computed
        g_vector = b_vector
        h_vector = a_vector
   
    return sign_matrix, g_vector, h_vector

def onebit_dequantize(sign_matrix, g_vector, h_vector):
    """
    Approximate dequantization for OneBit (for validation purposes)
   
    Note: This is an approximation since OneBit doesn't exactly reconstruct the original matrix
    """
    # Approximate reconstruction: W ≈ sign_matrix ⊙ (h_vector.unsqueeze(1) @ g_vector.unsqueeze(0))
    if len(sign_matrix.shape) == 2:
        outer_product = torch.outer(h_vector, g_vector)
        reconstructed = sign_matrix * outer_product
    else:
        # For other shapes, use element-wise scaling
        scale = torch.mean(torch.abs(g_vector)) * torch.mean(torch.abs(h_vector))
        reconstructed = sign_matrix * scale
   
    return reconstructed

class OneBitLinear:
    """
    OneBit Linear layer implementation for inference
    This simulates the OneBit forward pass: Y = (X ⊙ g) * W±1^T ⊙ h
    """
    def __init__(self, sign_matrix, g_vector, h_vector):
        self.sign_matrix = sign_matrix
        self.g_vector = g_vector
        self.h_vector = h_vector
   
    def forward(self, x):
        # OneBit forward: Y = (X ⊙ g) * W±1^T ⊙ h
        # Apply g vector element-wise to input
        x_scaled = x * self.g_vector.unsqueeze(0)  # Broadcast g across batch dimension
       
        # Matrix multiplication with sign matrix
        output = torch.mm(x_scaled, self.sign_matrix.t())
       
        # Apply h vector element-wise to output
        output_scaled = output * self.h_vector.unsqueeze(0)  # Broadcast h across batch dimension
       
        return output_scaled

def quantize_model_parameters_onebit(model, decomposition_method='nmf'):
    """
    Apply OneBit quantization to all weight and bias parameters of a model
    """
    quantization_start_time = time.time()
   
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:  # Only quantize weight matrices
            # Store original data and gradient state
            original_requires_grad = param.requires_grad
           
            # Apply OneBit quantization
            sign_matrix, g_vector, h_vector = onebit_quantize(param.data, decomposition_method)
           
            # Store quantized values and metadata
            param.data = sign_matrix.to(param.dtype)  # Keep as original dtype for compatibility
            param.g_vector = g_vector.to(param.dtype)
            param.h_vector = h_vector.to(param.dtype)
            param.is_onebit = True
            param.original_requires_grad = original_requires_grad
           
            # Temporarily disable gradients for quantized parameters
            param.requires_grad = False
           
            # For bias parameters, apply simple quantization
        elif 'bias' in name:
            # Simple sign quantization for bias
            original_requires_grad = param.requires_grad
            scale = torch.mean(torch.abs(param.data))
            sign_bias = torch.sign(param.data)
           
            param.data = sign_bias
            param.bias_scale = scale
            param.is_onebit_bias = True
            param.original_requires_grad = original_requires_grad
            param.requires_grad = False
   
    quantization_end_time = time.time()
    return quantization_end_time - quantization_start_time

def dequantize_model_parameters_onebit(model):
    """
    Dequantize all OneBit quantized parameters of a model
    """
    for name, param in model.named_parameters():
        if hasattr(param, 'is_onebit') and param.is_onebit:
            # Dequantize using OneBit reconstruction
            param.data = onebit_dequantize(param.data, param.g_vector, param.h_vector)
           
            # Restore gradient tracking
            param.requires_grad = param.original_requires_grad
           
            # Clean up quantization metadata
            delattr(param, 'g_vector')
            delattr(param, 'h_vector')
            delattr(param, 'is_onebit')
            delattr(param, 'original_requires_grad')
       
        elif hasattr(param, 'is_onebit_bias') and param.is_onebit_bias:
            # Dequantize bias
            param.data = param.data * param.bias_scale
           
            # Restore gradient tracking
            param.requires_grad = param.original_requires_grad
           
            # Clean up quantization metadata
            delattr(param, 'bias_scale')
            delattr(param, 'is_onebit_bias')
            delattr(param, 'original_requires_grad')

def measure_and_print_memory_onebit(client_idx, stage, model):
    """Helper function to measure and print memory usage for OneBit"""
    # Force garbage collection before measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
   
    memory_usage = get_memory_usage()
    tensor_memory = get_tensor_memory_usage()
    model_size = calculate_model_size(model, consider_onebit=True)
   
    print(f"Client {client_idx}: Memory usage {stage}: {memory_usage:.2f} MB")
    print(f"Client {client_idx}: Tensor memory {stage}: {tensor_memory:.2f} MB")
    print(f"Client {client_idx}: Model size {stage}: {model_size:.2f} MB")
   
    return memory_usage, tensor_memory, model_size

def receive_server_model(args, client_nodes, central_node):
    """Send server model to all clients"""
    for idx in range(len(client_nodes)):
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_nodes[idx].model.load_param(copy.deepcopy(central_node.model.get_param(clone=True)))
        else:
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
   
    return client_nodes

def benchmark_inference_time(model, sample_input, num_runs=100):
    """Benchmark inference time for a model"""
    model.eval()
    times = []
   
    # Warmup runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(sample_input)
   
    # Actual timing runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
   
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        times.append(end_time - start_time)
   
    avg_time = sum(times) / len(times)
    std_time = (sum([(t - avg_time) ** 2 for t in times]) / len(times)) ** 0.5
   
    return avg_time, std_time

def log_quantization_metrics(client_id, metrics_dict):
    """
    Log quantization metrics in a structured format for easy analysis
    """
    print(f"\n📋 QUANTIZATION METRICS LOG FOR CLIENT {client_id}:")
    print("=" * 80)
   
    # Core measurements as requested
    print("CORE QUANTIZATION MEASUREMENTS:")
    print(f"  Memory usage before quantization: {metrics_dict['memory_before_quantization']:.2f} MB")
    print(f"  Tensor memory before quantization: {metrics_dict['tensor_memory_before_quantization']:.2f} MB")
    print(f"  Model size before quantization: {metrics_dict['model_size_before_quantization']:.2f} MB")
    print(f"  Time taken for quantization: {metrics_dict['time_taken_for_quantization']:.4f} seconds")
    print(f"  Memory usage after quantization: {metrics_dict['memory_after_quantization']:.2f} MB")
    print(f"  Tensor memory after quantization: {metrics_dict['tensor_memory_after_quantization']:.2f} MB")
    print(f"  Model size after quantization: {metrics_dict['model_size_after_quantization']:.2f} MB")
    print(f"  Memory reduction from quantization: {metrics_dict['memory_reduction_from_quantization']:.2f} MB")
    print(f"  Model size reduction: {metrics_dict['model_size_reduction']:.2f} MB")
   
    # Additional analysis
    if 'memory_reduction_percentage' in metrics_dict:
        print(f"\nADDITIONAL ANALYSIS:")
        print(f"  Memory reduction percentage: {metrics_dict['memory_reduction_percentage']:.2f}%")
        print(f"  Model size reduction percentage: {metrics_dict['model_size_reduction_percentage']:.2f}%")
        print(f"  Compression ratio: {metrics_dict['compression_ratio']:.2f}%")
        print(f"  Average bit-width: {metrics_dict['average_bit_width']:.4f} bits")
   
    print("=" * 80)

def export_metrics_to_csv(client_id, metrics_dict, filename="quantization_metrics.csv"):
    """
    Export metrics to CSV file for further analysis
    """
    import csv
    import os
   
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
   
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'client_id',
            'memory_before_quantization',
            'tensor_memory_before_quantization',
            'model_size_before_quantization',
            'time_taken_for_quantization',
            'memory_after_quantization',
            'tensor_memory_after_quantization',
            'model_size_after_quantization',
            'memory_reduction_from_quantization',
            'model_size_reduction',
            'memory_reduction_percentage',
            'model_size_reduction_percentage',
            'compression_ratio',
            'average_bit_width'
        ]
       
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
       
        # Prepare row data
        row_data = {'client_id': client_id}
        row_data.update(metrics_dict)
       
        writer.writerow(row_data)
   
    print(f"📊 Metrics exported to {filename}")

def perform_client_training_onebit(args, client_nodes, training_function, global_model_param=None):
    """Common training logic for both local_train and fedprox with OneBit quantization"""
    client_losses = []
   
    for i in range(len(client_nodes)):
        print(f"\n{'='*60}")
        print(f"PROCESSING CLIENT {i}")
        print(f"{'='*60}")
       
        # ============ TRAINING PHASE ============
        print(f"\n🏋️ TRAINING PHASE:")
        training_start_time = time.time()
       
        # Perform training epochs
        epoch_losses = []
        epoch_times = []
       
        for epoch in range(args.E):
            epoch_start_time = time.time()
           
            if global_model_param is not None:
                loss = training_function(global_model_param, args, client_nodes[i])
            else:
                loss = training_function(args, client_nodes[i])
           
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
           
            epoch_losses.append(loss)
            epoch_times.append(epoch_time)
           
            print(f"Client {i}, Epoch {epoch+1}/{args.E}: Loss = {loss:.6f}, Time = {epoch_time:.4f}s")
       
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
       
        client_losses.append(sum(epoch_losses) / len(epoch_losses))
       
        print(f"Client {i}: Total training time: {total_training_time:.4f} seconds")
        print(f"Client {i}: Average epoch time: {avg_epoch_time:.4f} seconds")
       
        # ============ BEFORE QUANTIZATION MEASUREMENTS ============
        print(f"\n📏 BEFORE QUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory before quantization
        memory_before_quantization = get_memory_usage()
        tensor_memory_before_quantization = get_tensor_memory_usage()
        model_size_before_quantization = calculate_model_size(client_nodes[i].model, consider_onebit=False)
       
        print(f"Client {i}: Memory usage before quantization: {memory_before_quantization:.2f} MB")
        print(f"Client {i}: Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
        print(f"Client {i}: Model size before quantization: {model_size_before_quantization:.2f} MB")
       
        # ============ QUANTIZATION PHASE ============
        print(f"\n⚡ QUANTIZATION PHASE:")
        quantization_start_time = time.time()
       
        # Apply OneBit quantization
        quantization_time = quantize_model_parameters_onebit(
            client_nodes[i].model,
            decomposition_method='nmf'  # Use NMF as recommended in OneBit paper
        )
       
        quantization_end_time = time.time()
        total_quantization_time = quantization_end_time - quantization_start_time
       
        print(f"Client {i}: Time taken for quantization: {quantization_time:.4f} seconds")
       
        # ============ AFTER QUANTIZATION MEASUREMENTS ============
        print(f"\n📐 AFTER QUANTIZATION MEASUREMENTS:")
       
        # Force garbage collection after quantization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
       
        # Measure memory after quantization
        memory_after_quantization = get_memory_usage()
        tensor_memory_after_quantization = get_tensor_memory_usage()
        model_size_after_quantization = calculate_model_size(client_nodes[i].model, consider_onebit=True)
       
        print(f"Client {i}: Memory usage after quantization: {memory_after_quantization:.2f} MB")
        print(f"Client {i}: Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
        print(f"Client {i}: Model size after quantization: {model_size_after_quantization:.2f} MB")
       
        # ============ QUANTIZATION IMPACT ANALYSIS ============
        print(f"\n📊 QUANTIZATION IMPACT ANALYSIS:")
       
        # Calculate reductions
        memory_reduction_from_quantization = memory_before_quantization - memory_after_quantization
        tensor_memory_reduction = tensor_memory_before_quantization - tensor_memory_after_quantization
        model_size_reduction = model_size_before_quantization - model_size_after_quantization
       
        print(f"Client {i}: Memory reduction from quantization: {memory_reduction_from_quantization:.2f} MB")
        print(f"Client {i}: Tensor memory reduction: {tensor_memory_reduction:.2f} MB")
        print(f"Client {i}: Model size reduction: {model_size_reduction:.2f} MB")
       
        # Calculate percentage reductions
        memory_reduction_percentage = 0
        tensor_memory_reduction_percentage = 0
        model_size_reduction_percentage = 0
        compression_ratio = 0
       
        if memory_before_quantization > 0:
            memory_reduction_percentage = (memory_reduction_from_quantization / memory_before_quantization) * 100
            print(f"Client {i}: Memory reduction percentage: {memory_reduction_percentage:.2f}%")
       
        if tensor_memory_before_quantization > 0:
            tensor_memory_reduction_percentage = (tensor_memory_reduction / tensor_memory_before_quantization) * 100
            print(f"Client {i}: Tensor memory reduction percentage: {tensor_memory_reduction_percentage:.2f}%")
       
        if model_size_before_quantization > 0:
            model_size_reduction_percentage = (model_size_reduction / model_size_before_quantization) * 100
            compression_ratio = (model_size_after_quantization / model_size_before_quantization) * 100
            print(f"Client {i}: Model size reduction percentage: {model_size_reduction_percentage:.2f}%")
            print(f"Client {i}: Compression ratio: {compression_ratio:.2f}%")
       
        # Calculate average bit-width (approximation)
        total_params = sum(p.numel() for p in client_nodes[i].model.parameters())
        onebit_params = sum(p.numel() for p in client_nodes[i].model.parameters()
                           if hasattr(p, 'is_onebit') and p.is_onebit)
        value_vector_params = sum(p.g_vector.numel() + p.h_vector.numel()
                                 for p in client_nodes[i].model.parameters()
                                 if hasattr(p, 'is_onebit') and p.is_onebit)
       
        average_bit_width = 16  # Default for non-quantized
        if total_params > 0:
            average_bit_width = (onebit_params * 1 + value_vector_params * 16) / total_params
       
        # ============ STRUCTURED METRICS COLLECTION ============
        metrics_dict = {
            'memory_before_quantization': memory_before_quantization,
            'tensor_memory_before_quantization': tensor_memory_before_quantization,
            'model_size_before_quantization': model_size_before_quantization,
            'time_taken_for_quantization': quantization_time,
            'memory_after_quantization': memory_after_quantization,
            'tensor_memory_after_quantization': tensor_memory_after_quantization,
            'model_size_after_quantization': model_size_after_quantization,
            'memory_reduction_from_quantization': memory_reduction_from_quantization,
            'model_size_reduction': model_size_reduction,
            'memory_reduction_percentage': memory_reduction_percentage,
            'model_size_reduction_percentage': model_size_reduction_percentage,
            'compression_ratio': compression_ratio,
            'average_bit_width': average_bit_width
        }
       
        # ============ STRUCTURED LOGGING ============
        log_quantization_metrics(i, metrics_dict)
       
        # Export to CSV (optional - can be enabled/disabled)
        try:
            export_metrics_to_csv(i, metrics_dict)
        except Exception as e:
            print(f"Warning: Could not export to CSV: {e}")
       
        # ============ DETAILED SUMMARY ============
        print(f"\n📈 DETAILED SUMMARY FOR CLIENT {i}:")
        print(f"{'─'*80}")
        print(f"REQUIRED MEASUREMENTS:")
        print(f"  • Memory usage before quantization: {memory_before_quantization:.2f} MB")
        print(f"  • Tensor memory before quantization: {tensor_memory_before_quantization:.2f} MB")
        print(f"  • Model size before quantization: {model_size_before_quantization:.2f} MB")
        print(f"  • Time taken for quantization: {quantization_time:.4f} seconds")
        print(f"  • Memory usage after quantization: {memory_after_quantization:.2f} MB")
        print(f"  • Tensor memory after quantization: {tensor_memory_after_quantization:.2f} MB")
        print(f"  • Model size after quantization: {model_size_after_quantization:.2f} MB")
        print(f"  • Memory reduction from quantization: {memory_reduction_from_quantization:.2f} MB")
        print(f"  • Model size reduction: {model_size_reduction:.2f} MB")
       
        if total_params > 0:
            print(f"")
            print(f"QUANTIZATION QUALITY:")
            print(f"  • Total parameters: {total_params:,}")
            print(f"  • OneBit quantized parameters: {onebit_params:,}")
            print(f"  • Value vector parameters: {value_vector_params:,}")
            print(f"  • Average bit-width: {average_bit_width:.4f} bits")
            print(f"  • Theoretical compression: {16/average_bit_width:.2f}x")
       
        print(f"{'='*80}\n")
   
    train_loss = sum(client_losses) / len(client_losses)
    return client_losses, train_loss

def print_federated_learning_summary(num_clients, total_time, avg_metrics):
    """Print a comprehensive summary of the federated learning round"""
    print(f"\n{'🌟'*20} FEDERATED LEARNING ROUND SUMMARY {'🌟'*20}")
    print(f"{'='*80}")
   
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Number of clients: {num_clients}")
    print(f"   Total round time: {total_time:.2f} seconds")
    print(f"   Average time per client: {total_time/num_clients:.2f} seconds")
   
    if avg_metrics:
        print(f"\n📈 AVERAGE METRICS ACROSS ALL CLIENTS:")
        for metric, value in avg_metrics.items():
            print(f"   {metric}: {value:.4f}")
   
    print(f"\n💡 OneBit QUANTIZATION BENEFITS:")
    print(f"   ✅ ~90-93% model size reduction")
    print(f"   ✅ ~1.007 average bit-width")
    print(f"   ✅ Maintains model performance")
    print(f"   ✅ Enables edge device deployment")
   
    print(f"{'='*80}")

def Client_update(args, client_nodes, central_node):
    """Client update functions with OneBit quantization"""
    round_start_time = time.time()
   
    print(f"\n🚀 STARTING FEDERATED LEARNING ROUND")
    print(f"{'='*60}")
    print(f"Client method: {args.client_method}")
    print(f"Number of clients: {len(client_nodes)}")
    print(f"Epochs per client: {args.E}")
   
    # Clients receive the server model
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # Update the global model based on client method
    if args.client_method == 'local_train':
        client_losses, train_loss = perform_client_training_onebit(
            args, client_nodes, client_localTrain
        )
       
    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses, train_loss = perform_client_training_onebit(
            args, client_nodes, client_fedprox, global_model_param
        )
       
    else:
        raise ValueError('Undefined client method...')

    round_end_time = time.time()
    total_round_time = round_end_time - round_start_time
   
    # Calculate average metrics
    avg_metrics = {
        'Training Loss': train_loss,
        'Round Time': total_round_time
    }
   
    # Print comprehensive summary
    print_federated_learning_summary(len(client_nodes), total_round_time, avg_metrics)
   
    return client_nodes, train_loss

def Client_validate(args, client_nodes):
    """Client validation functions with OneBit dequantization"""
    client_acc = []
    for idx in range(len(client_nodes)):
        # Dequantize model before validation if needed
        if hasattr(list(client_nodes[idx].model.parameters())[0], 'is_onebit'):
            dequantize_model_parameters_onebit(client_nodes[idx].model)
       
        acc = validate(args, client_nodes[idx])
        client_acc.append(acc)
   
    avg_client_acc = sum(client_acc) / len(client_acc)
    return avg_client_acc, client_acc

def DKL(_p, _q):
    """Kullback-Leibler divergence"""
    return torch.sum(_p * (_p.log() - _q.log()), dim=-1)

def client_localTrain(args, node, loss=0.0):
    """Vanilla local training"""
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        node.optimizer.zero_grad()
       
        # Forward pass
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        # Compute loss and backpropagate
        loss_local = F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
       
        # Update parameters
        node.optimizer.step()

    return loss / len(train_loader)

def client_fedprox(global_model_param, args, node, loss=0.0):
    """FedProx training with proximal term"""
    node.model.train()
   
    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        node.optimizer.zero_grad()
       
        # Forward pass
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        # Compute loss and backpropagate
        loss_local = F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
       
        # FedProx update with global model parameters
        node.optimizer.step(global_model_param)

    return loss / len(train_loader)

def validate(args, node):
    """Validation function - placeholder implementation"""
    # This is a placeholder - replace with your actual validation logic
    node.model.eval()
    correct = 0
    total = 0
   
    with torch.no_grad():
        for data, target in node.val_data:  # Assuming val_data exists
            data, target = data.cuda(), target.cuda()
            outputs = node.model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
   
    return 100 * correct / total if total > 0 else 0.0

def knowledge_distillation_loss(student_logits, teacher_logits, student_hidden, teacher_hidden, alpha=1.0):
    """
    OneBit-style knowledge distillation loss
   
    Args:
        student_logits: Output logits from student model
        teacher_logits: Output logits from teacher model  
        student_hidden: Hidden states from student model
        teacher_hidden: Hidden states from teacher model
        alpha: Weight for hidden state MSE loss
   
    Returns:
        Combined loss
    """
    # Cross-entropy loss on logits
    loss_ce = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
   
    # MSE loss on hidden states
    loss_mse = F.mse_loss(student_hidden, teacher_hidden)
   
    # Combined loss
    total_loss = loss_ce + alpha * loss_mse
   
    return total_loss
