from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
args = args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
import time
from collections import Counter

if __name__ == '__main__':

    
    # Set random seeds
    setup_seed(args.random_seed)
    print(args)

 

    # Loading data
    data = Data(args)

  
    # Data-size-based aggregation weights
    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    
    # Initialize the central node
    # num_id equals to -1 stands for central node
    central_node = Node(-1, data.test_loader[0], data.test_set, args)
 
    # Initialize the client nodes
    client_nodes = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
    
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []
    avgtime=[]
    
    for rounds in range(args.T):
  
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_nodes, args)

        client_nodes, train_loss = Client_update(args, client_nodes, central_node)
     
        avg_client_acc,client_acc = Client_validate(args, client_nodes)
        print(args.server_method + args.client_method + ', averaged clients personalization acc is ', avg_client_acc)
        

        
        # Partial select function
        if args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(client_nodes))]
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)
 
        # Server update
        start = time.perf_counter() 
     

        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,rounds_num=rounds)
        
        end = time.perf_counter() 

        
        acc = validate(args, central_node, which_dataset = 'local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)


        

        print('Running time: %s Seconds' % (end - start))
        avgtime.append(end - start)
        best_acc = max(test_acc_recorder)
        print("Current_Best test acc is:", best_acc)

        # Final acc recorder
        if rounds >= args.T - 10:
            final_test_acc_recorder.update(acc)

    
    print("Avg runing time:",np.mean(avgtime))

    # best acc
    best_acc = max(test_acc_recorder)
    print("Final_Best test acc is:", best_acc)

    # best top 10
    top_10_acc = sorted(test_acc_recorder, reverse=True)[:10]
    top_10_avg = np.mean(top_10_acc)
    top_10_std = np.std(top_10_acc)
    print("Top 10 test acc avg is:", top_10_avg)
    print("Top 10 test acc std is:", top_10_std)