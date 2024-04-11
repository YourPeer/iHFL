from ..StandardFL.Server import Server
import torch.distributed as dist
import sys
import multiprocessing as mp
from ..tools import *
class nebual_gateway(Server):
    def __init__(self, gateway_id, server_id, args, distributer, model):
        super().__init__(server_id,args, distributer, model)
        self.gateway_id = gateway_id
        self.server_id = server_id
        self.topology=args.topology
        self.leaders_num=len(args.topology[self.gateway_id])
        self.gateway_rounds=args.gateway_rounds
        # send extra info
        self.alpha=args.async_alpha
        self.staleness_func = args.staleness_func
        self.T = 0
        self.info = extra_info(self.global_loss, self.gateway_id, self.T)
        self.leaders_list = [l[0] for l in self.topology[self.gateway_id]]


    def init_process(self):
        setup_seed(2024)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.gateway_id, world_size=self.size)

    def run(self):
        self.init_process()
        lock = mp.Lock()

        for r in range(self.rounds):
            should_stop = False
            for gr in range(self.gateway_rounds):
                lock.acquire()
                if self.gateway_rounds - gr - 1 < self.leaders_num:
                    should_stop= True
                self.async_aggregation_with_leader(should_stop)  # aggregation gateway model
                lock.release()
            self.info = extra_info(self.global_loss, self.gateway_id, self.T)
            try:
                self.aggregation_with_cloud()# aggregation server model
            except:
                sys.exit()
            self.T+=1

    def aggregation_with_cloud(self):
        gateway_weight_vec, info_len = pack(self.model, self.info)
        dist.send(gateway_weight_vec, self.server_id)
        dist.recv(gateway_weight_vec,self.server_id)
        for i in self.leaders_list:
            dist.send(gateway_weight_vec,i)
        gateway_weight, _ = unpack(gateway_weight_vec, info_len)
        load_weights(self.model, gateway_weight)

    def async_aggregation_with_leader(self,should_stop):
        gateway_weight_vec, info_len = pack(self.model, self.info)
        leader_weights_vec=torch.zeros_like(gateway_weight_vec)
        dist.recv(leader_weights_vec)
        gateway_weights, _ = unpack(gateway_weight_vec, info_len)
        leader_weights, (leader_trainloss, leader_id, leader_tao) = unpack(leader_weights_vec, info_len)
        gateway_weights = self.staleness_aggregation(gateway_weights, leader_weights, leader_tao)
        load_weights(self.model, gateway_weights)
        gateway_weights_vec, info_len = pack(self.model, self.info)
        if not should_stop:
            dist.send(gateway_weights_vec, leader_id)


    def selection(self):
        if self.select_type=="random":
            client_num=len(self.clients_list)
            sampled_clients = np.zeros(client_num, dtype=int)
            selected_clients_idx = np.random.choice(
                range(0, client_num), int(client_num*self.select_ratio), replace=False
            )
            selected_clients_idx=sorted(selected_clients_idx)
            sampled_clients[selected_clients_idx]=1  #[0,0,1,0,1,0...]
            selected_clients=[i for i,j in zip(self.clients_list,sampled_clients) if j==1]
            data_ratio = [ratio/np.sum(self.data_ratio[selected_clients]) for ratio in self.data_ratio]
        return sampled_clients,data_ratio

    def download_global_model(self,server_id):
        weights_vec, info_len = pack(self.model, self.info)
        dist.recv(weights_vec, server_id)
        weights, info = unpack(weights_vec, info_len)
        load_weights(self.model, weights)

    def borcast_model_to_leader(self):
        # borcast edge model
        global_weight_vec, info_len = pack(self.model, self.info)
        for i in self.leaders_list:
            dist.send(global_weight_vec, i)

    def staleness_aggregation(self, global_weights, gateway_weights,gateway_tao):
        staleness=self.T-gateway_tao
        alpha_t = self.alpha * self.staleness(staleness)
        global_weights = (1-alpha_t) * global_weights + alpha_t * gateway_weights
        return global_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "poly":
            a = 0.3
            return pow(staleness+1, -a)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)