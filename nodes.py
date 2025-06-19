import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import DatasetSplit
from utils import init_model
from utils import init_optimizer, model_parameter_vector


class Node(object):
    def __init__(self, num_id, local_data, train_set, args):
        try:
            self.num_id = num_id
            self.args = args
            self.node_num = self.args.node_num
            
            if num_id == -1:
                self.valid_ratio = args.server_valid_ratio
            else:
                self.valid_ratio = args.client_valid_ratio

            if self.args.dataset == 'cifar10' or self.args.dataset == 'fmnist':
                self.num_classes = 10
            elif self.args.dataset == 'cifar100':
                self.num_classes = 100
            elif self.args.dataset == 'tinyimagenet':
                self.num_classes = 200
            else:
                self.num_classes = 10  # Default fallback

            # Data splitting
            if args.iid == 1 or num_id == -1:
                # for the server, use the validate_set as the training data, and use local_data for testing
                self.local_data, self.validate_set = self.train_val_split_forServer(
                    local_data.indices, train_set, self.valid_ratio, self.num_classes
                )
            else:
                self.local_data, self.validate_set = self.train_val_split(
                    local_data, train_set, self.valid_ratio
                )
            
            # Model initialization with error handling
            try:
                self.model = init_model(self.args.local_model, self.args).cuda()
                print(f"✅ Successfully initialized model for node {num_id}")
            except Exception as e:
                print(f"❌ Error initializing model for node {num_id}: {e}")
                raise
            
            # Optimizer initialization with error handling
            try:
                self.optimizer = init_optimizer(self.num_id, self.model, args)
                print(f"✅ Successfully initialized optimizer for node {num_id}")
            except Exception as e:
                print(f"❌ Error initializing optimizer for node {num_id}: {e}")
                # Create a default optimizer as fallback
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            
            # Node init for feddyn
            if hasattr(args, 'client_method') and args.client_method == 'feddyn':
                try:
                    self.old_grad = None
                    self.old_grad = copy.deepcopy(self.model)
                    self.old_grad = model_parameter_vector(args, self.old_grad)
                    self.old_grad = torch.zeros_like(self.old_grad)
                except Exception as e:
                    print(f"Warning: Error initializing FedDyn components for node {num_id}: {e}")
                    self.old_grad = None
                    
            if hasattr(args, 'server_method') and 'feddyn' in args.server_method:
                try:
                    self.server_state = copy.deepcopy(self.model)
                    for param in self.server_state.parameters():
                        param.data = torch.zeros_like(param.data)
                except Exception as e:
                    print(f"Warning: Error initializing FedDyn server state for node {num_id}: {e}")
                    self.server_state = None
            
            # Node init for fedadam's server
            if hasattr(args, 'server_method') and args.server_method == 'fedadam' and num_id == -1:
                try:
                    m = copy.deepcopy(self.model)
                    self.zero_weights(m)
                    self.m = m
                    v = copy.deepcopy(self.model)
                    self.zero_weights(v)
                    self.v = v
                except Exception as e:
                    print(f"Warning: Error initializing FedAdam components for server: {e}")
                    self.m = None
                    self.v = None
                    
        except Exception as e:
            print(f"Critical error initializing Node {num_id}: {e}")
            raise

    def zero_weights(self, model):
        """Zero out all model weights"""
        try:
            for n, p in model.named_parameters():
                p.data.zero_()
        except Exception as e:
            print(f"Error zeroing weights: {e}")

    def train_val_split(self, idxs, train_set, valid_ratio): 
        """Split training data into train and validation sets"""
        try:
            np.random.shuffle(idxs)

            validate_size = valid_ratio * len(idxs)
            idxs_test = idxs[:int(validate_size)]
            idxs_train = idxs[int(validate_size):]

            train_loader = DataLoader(
                DatasetSplit(train_set, idxs_train),
                batch_size=self.args.batchsize, 
                num_workers=0, 
                shuffle=True
            )

            test_loader = DataLoader(
                DatasetSplit(train_set, idxs_test),
                batch_size=self.args.validate_batchsize,  
                num_workers=0, 
                shuffle=True
            )
            
            return train_loader, test_loader
        except Exception as e:
            print(f"Error in train_val_split: {e}")
            # Return empty loaders as fallback
            empty_dataset = DatasetSplit(train_set, [])
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10):
        """Split data for server with balanced classes"""
        try:
            np.random.shuffle(idxs)
            
            validate_size = int(valid_ratio * len(idxs))

            # Generate proxy dataset with balanced classes
            idxs_test = []

            if hasattr(self.args, 'longtail_proxyset') and self.args.longtail_proxyset == 'none':
                test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]
            elif hasattr(self.args, 'longtail_proxyset') and self.args.longtail_proxyset == 'LT':
                imb_factor = 0.1
                test_class_count = [
                    int(validate_size/num_classes * (imb_factor**(_classes_idx / (num_classes - 1.0)))) 
                    for _classes_idx in range(num_classes)
                ]
            else:
                # Default balanced distribution
                test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]

            k = 0
            while sum(test_class_count) != 0 and k < len(idxs):
                try:
                    label = train_set[idxs[k]][1]
                    if test_class_count[label] > 0:
                        idxs_test.append(idxs[k])
                        test_class_count[label] -= 1
                except Exception as e:
                    print(f"Warning: Error processing sample {k}: {e}")
                k += 1
                
            # Collect labels for verification
            label_list = []
            for k in idxs_test:
                try:
                    label_list.append(train_set[k][1])
                except Exception as e:
                    print(f"Warning: Error getting label for sample {k}: {e}")

            idxs_train = [idx for idx in idxs if idx not in idxs_test]

            train_loader = DataLoader(
                DatasetSplit(train_set, idxs_train),
                batch_size=self.args.batchsize, 
                num_workers=0, 
                shuffle=True
            )
            
            test_loader = DataLoader(
                DatasetSplit(train_set, idxs_test),
                batch_size=self.args.validate_batchsize,  
                num_workers=0, 
                shuffle=True
            )

            return train_loader, test_loader
        except Exception as e:
            print(f"Error in train_val_split_forServer: {e}")
            # Return empty loaders as fallback
            empty_dataset = DatasetSplit(train_set, [])
            empty_loader = DataLoader(empty_dataset, batch_size=1, num_workers=0, shuffle=False)
            return empty_loader, empty_loader


# Tools for long-tailed functions
def label_indices2indices(list_label2indices):
    """Convert list of label indices to flat indices list"""
    try:
        indices_res = []
        for indices in list_label2indices:
            indices_res.extend(indices)
        return indices_res
    except Exception as e:
        print(f"Error in label_indices2indices: {e}")
        return []

def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    """Calculate number of images per class for long-tail distribution"""
    try:
        img_max = len(list_label2indices_train) / num_classes
        img_num_per_cls = []
        
        if imb_type == 'exp':
            for _classes_idx in range(num_classes):
                num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            # Default uniform distribution
            for _classes_idx in range(num_classes):
                img_num_per_cls.append(int(img_max))

        return img_num_per_cls
    except Exception as e:
        print(f"Error in _get_img_num_per_cls: {e}")
        # Return uniform distribution as fallback
        img_max = len(list_label2indices_train) / num_classes if list_label2indices_train else 100
        return [int(img_max) for _ in range(num_classes)]

def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    """Generate long-tail training distribution"""
    try:
        new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
        img_num_list = _get_img_num_per_cls(
            copy.deepcopy(new_list_label2indices_train), 
            num_classes, 
            imb_factor, 
            imb_type
        )
        
        print('img_num_class')
        print(img_num_list)

        list_clients_indices = []
        classes = list(range(num_classes))
        
        for _class, _img_num in zip(classes, img_num_list):
            try:
                indices = list_label2indices_train[_class]
                np.random.shuffle(indices)
                idx = indices[:_img_num]
                list_clients_indices.append(idx)
            except Exception as e:
                print(f"Warning: Error processing class {_class}: {e}")
                list_clients_indices.append([])
                
        num_list_clients_indices = label_indices2indices(list_clients_indices)
        print('All num_data_train')
        print(len(num_list_clients_indices))

        return img_num_list, list_clients_indices
    except Exception as e:
        print(f"Error in train_long_tail: {e}")
        # Return default values
        default_img_num = [100] * num_classes
        default_indices = [[] for _ in range(num_classes)]
        return default_img_num, default_indices

