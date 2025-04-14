
import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math

### basic functions for models
# def init_weights(net, state):
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
                module.running_mean = param_dict[name + '.running_mean']
                module.running_var = param_dict[name + '.running_var']

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w.to(self.flat_w.device))
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def reshape_flat_weights(self, flat_w):
       
        reshaped_weights = {
            f"{m}.{n}": t.view(s)
            for (m, n), t, s in zip(self._weights_module_names, flat_w.split(self._weights_numels), self._weights_shapes)
        }
        return reshaped_weights
   
    def get_head_weights(self, flat_w):
       
        reshaped_weights = self.reshape_flat_weights(flat_w)
        linear_weights = [
            weight.view(-1)
            for name, weight in reshaped_weights.items()
            if 'Linear' in name
        ]
        if linear_weights:
            return torch.cat(linear_weights)
        else:
            return torch.tensor([]) 

    def get_body_weights(self, flat_w):
      
        reshaped_weights = self.reshape_flat_weights(flat_w)
        non_linear_weights = [
            weight.view(-1)
            for name, weight in reshaped_weights.items()
            if 'Linear' not in name
        ]
        if non_linear_weights:
            return torch.cat(non_linear_weights)
        else:
            return torch.tensor([])


    def forward_with_param(self, inp, new_w, *args, **kwargs):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        if 'flat_w' in state_dict:
            self.flat_w.data = state_dict['flat_w'].detach().clone().requires_grad_(True).to(self.flat_w.device)

            state_dict = {k: v for k, v in state_dict.items() if k != 'flat_w'}
            super(ReparamModule, self).load_state_dict(state_dict, strict)

            ws = (t.view(s) for (t, s) in zip(self.flat_w.split(self._weights_numels), self._weights_shapes))
            for (m, n), w in zip(self._weights_module_names, ws):
                setattr(m, n, w.to(self.flat_w.device))
        else:
            super(ReparamModule, self).load_state_dict(state_dict, strict)

    def __call__(self, inp, *args, **kwargs):
        return self.forward_with_param(inp, self.flat_w, *args, **kwargs)