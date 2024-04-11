from FLTask import generated_task
import argparse
import multiprocessing as mp
from Frameworks import nebula_server,nebual_gateway,nebula_client
def get_args():
    parser = argparse.ArgumentParser()
    # System parameters
    parser.add_argument('--clients', type=int, default=12)
    parser.add_argument('--gateways', type=int, default=3)
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--port', type=int, default=2024)

    # Task parameters
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--model_name', type=str, default='cifarcnn')
    parser.add_argument('--data_type', type=str, default='iid')  # iid,niid,sharding_max
    parser.add_argument('--partition_dir', type=float, default=0.6)
    parser.add_argument('--partition_shards', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--gateway_rounds', type=int, default=10)
    parser.add_argument('--local_steps', type=int, default=100)

    # Training  parameters
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--select_type', type=str, default="random")
    parser.add_argument('--select_ratio', type=float, default=0.5)
    parser.add_argument('--async_alpha', type=float, default=0.9)
    parser.add_argument('--staleness_func', type=str, default="constant") # constant, poly, hinge

    args = parser.parse_args()
    return args

def generate_topology(clients, gateways):
    return {i + clients: [list(range(i, clients+gateways, gateways))[j:j+2] for j in range(0, len(list(range(i, clients+gateways, gateways))) - 1, 2)] for i in range(gateways)}

if __name__=="__main__":
    args = get_args()
    args.size=args.clients+args.gateways+1
    # distributer: dict, key: global,local,data_map,num_classes
    args.topology = generate_topology(args.clients, args.gateways)
    print(args.topology)
    distributer,model=generated_task(args.data_path,args.dataset_name,args.model_name,args.clients,args.batchsize,args.data_type,args.partition_dir,args.partition_shards)
    processes = []
    for c in range(args.clients+args.gateways+1):
        print(c)
        # -------------------------start server--------------------------------
        if c == args.clients+args.gateways:
            print("server:%d start"%c)
            S = nebula_server(c, args,distributer,model)
            p = mp.Process(target=S.run, args=())
        # -------------------------start gateways--------------------------------
        elif c in args.topology.keys():
            print("gateway:%d start" % c)
            g=nebual_gateway(c, args.clients+args.gateways, args, distributer, model)
            p = mp.Process(target=g.run, args=())
        # -------------------------start clients--------------------------------
        else:
            print("client:%d start" % c)
            C = nebula_client(c, args,distributer,model)
            p = mp.Process(target=C.run, args=())
        p.start()
        processes.append(p)

    # stop training
    processes[-1].join()
    if not processes[-1].is_alive():
        for p in processes[:-1]:
            p.terminate()
