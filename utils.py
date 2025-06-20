import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import psutil
from torch.backends import cudnn
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn
from models_dict.vit import ViT, ViT_fedlaw

##############################################################################
# Basic Utility Functions
##############################################################################

def setup_seed(seed):
    """Setup random seeds for reproducibility"""
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
    except Exception as e:
        print(f"Warning: Error setting up seeds: {e}")

def generate_selectlist(client_node, ratio=0.5):
    """Generate list of selected clients"""
    try:
        candidate_list = [i for i in range(len(client_node))]
        select_num = int(ratio * len(client_node))
        select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
        return select_list
    except Exception as e:
        print(f"Error generating select list: {e}")
        return list(range(len(client_node)))

def lr_scheduler(rounds, node_list, args):
    """Learning rate scheduler"""
    try:
        if rounds != 0:
            args.lr *= 0.99
            for i in range(len(node_list)):
                node_list[i].args.lr = args.lr
                node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    except Exception as e:
        print(f"Warning: Error in lr_scheduler: {e}")

##############################################################################
# Model Initialization
##############################################################################

def init_model(model_type, args):
    """Initialize model based on type and dataset"""
    try:
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
            elif model_type == 'ResNet20':
                model = resnet.ResNet20_fedlaw(num_classes)
            elif model_type == 'ResNet18':
                model = resnet.ResNet18_fedlaw(num_classes)
            elif model_type == 'MLP':
                model = cnn.MLP_fedlaw()
            elif model_type == 'LeNet5':
                model = cnn.LeNet5_fedlaw()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            if model_type == 'CNN':
                if args.dataset == 'cifar10':
                    model = cnn.CNNCifar10()
                elif args.dataset == 'fmnist':
                    model = cnn.CNNfmnist()
                else:
                    model = cnn.CNNCifar100()
            elif model_type == 'ResNet20':
                model = resnet.ResNet20(num_classes)
            elif model_type == 'ResNet18':
                model = resnet.ResNet18(num_classes)
            elif model_type == 'MLP':
                model = cnn.MLP()
            elif model_type == 'LeNet5':
                model = cnn.LeNet5()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Set initial model_id as temporary - will be overridden by Node class
        if hasattr(model, 'model_id'):
            model.model_id = f'{model_type}_temp'

        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

def init_optimizer(num_id, model, args):
    """Initialize optimizer for the model"""
    try:
        if num_id > -1 and args.client_method == 'fedprox':
            optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        return optimizer
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return torch.optim.SGD(model.parameters(), lr=0.01)

##############################################################################
# Validation Functions
##############################################################################

def validate(args, node, which_dataset='validate'):
    """Enhanced validation function"""
    try:
        node.model.cuda().eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        correct = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
            
            if hasattr(test_loader, 'dataset'):
                acc = correct / len(test_loader.dataset) * 100
            else:
                acc = correct / total_samples * 100
        
        return acc
    except Exception as e:
        print(f"Error in validation: {e}")
        return 0.0

def testloss(args, node, which_dataset='validate'):
    """Test loss computation"""
    try:
        node.model.cuda().eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        loss = []
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model(data)
                loss_local = F.cross_entropy(output, target, reduction='mean')
                loss.append(loss_local.item())
                total_samples += target.size(0)
        
        loss_value = sum(loss) / len(loss) if len(loss) > 0 else 0.0
        return loss_value
    except Exception as e:
        print(f"Error in test loss computation: {e}")
        return 0.0

##############################################################################
# FedLAW Support Functions
##############################################################################

def validate_with_param(args, node, param, which_dataset='validate'):
    """FedLAW validation with parameters"""
    try:
        node.model.cuda().eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        correct = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model.forward_with_param(data, param)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
            
            if hasattr(test_loader, 'dataset'):
                acc = correct / len(test_loader.dataset) * 100
            else:
                acc = correct / total_samples * 100
        
        return acc
    except Exception as e:
        print(f"Error in validate_with_param: {e}")
        return 0.0

def testloss_with_param(args, node, param, which_dataset='validate'):
    """FedLAW test loss with parameters"""
    try:
        node.model.cuda().eval()
        
        if which_dataset == 'validate':
            test_loader = node.validate_set
        elif which_dataset == 'local':
            test_loader = node.local_data
        else:
            raise ValueError('Undefined dataset type')

        loss = []
        total_samples = 0
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = node.model.forward_with_param(data, param)
                loss_local = F.cross_entropy(output, target, reduction='mean')
                loss.append(loss_local.item())
                total_samples += target.size(0)
        
        loss_value = sum(loss) / len(loss) if len(loss) > 0 else 0.0
        return loss_value
    except Exception as e:
        print(f"Error in testloss_with_param: {e}")
        return 0.0

##############################################################################
# Optimizer Classes
##############################################################################

class PerturbedGradientDescent(Optimizer):
    """Perturbed Gradient Descent optimizer for FedProx"""
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
# Utility Classes
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps) if self.steps > 0 else 0

def model_parameter_vector(args, model):
    """Extract model parameters as a vector"""
    try:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            if hasattr(model, 'flat_w'):
                vector = model.flat_w
            else:
                param = [p.view(-1) for p in model.parameters()]
                vector = torch.cat(param, dim=0)
        else:
            param = [p.view(-1) for p in model.parameters()]
            vector = torch.cat(param, dim=0)
        return vector
    except Exception as e:
        print(f"Error extracting model parameter vector: {e}")
        return torch.tensor([])
