from FLTask import generated_task
import argparse
import multiprocessing as mp
from Frameworks import Sp_server,Sp_client
import warnings
warnings.filterwarnings("ignore")
def get_args():
    parser = argparse.ArgumentParser()
    # System parameters
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--port', type=int, default=2000)

    # Task parameters
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--model_name', type=str, default='cifarcnn')
    parser.add_argument('--data_type', type=str, default='niid')  # iid,niid,sharding_max
    parser.add_argument('--partition_dir', type=float, default=0.6)
    parser.add_argument('--partition_shards', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local_steps', type=int, default=100)


    # Training  parameters
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--select_type', type=str, default="random")
    parser.add_argument('--select_ratio', type=float, default=1)
    parser.add_argument('--async_alpha', type=float, default=0.6)

    # Sparse parameters
    parser.add_argument('--sparse', type=bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', type=bool, default=True,
                        help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--importanace_mode', type=str, default='weight',
                        help='sparse initialization')  # uniform/uniform_plus/ERK/ERK_plus/ER/snip/GraSP
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='The density of the overall sparse network.')
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_args()
    args.size = args.clients+1
    # distributer: dict, key: global,local,data_map,num_classes
    distributer,model=generated_task(args.data_path,args.dataset_name,args.model_name,args.clients,args.batchsize,args.data_type,args.partition_dir,args.partition_shards)
    processes = []
    for c in range(args.clients+1):
        if c == args.clients:
            # -------------------------start server--------------------------------
            S = Sp_server(c, args,distributer,model)
            p = mp.Process(target=S.run, args=())
        else:
            # -------------------------start client--------------------------------
            C = Sp_client(c, args,distributer,model)
            p = mp.Process(target=C.run, args=())
        p.start()
        processes.append(p)

