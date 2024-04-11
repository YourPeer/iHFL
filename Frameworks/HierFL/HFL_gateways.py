from ..StandardFL.Server import Server
import torch.distributed as dist
import sys
from ..tools import *
class Gateway(Server):
    def __init__(self, gateway_id, server_id, args, distributer, model):
        super().__init__(server_id,args, distributer, model)
        self.gateway_id = gateway_id
        self.server_id = server_id
        self.topology=args.topology
        self.clients_list = self.topology[self.gateway_id]
        self.gateway_rounds=args.gateway_rounds
        # send extra info
        self.T = 0
        self.info = extra_info(self.global_loss, self.gateway_id, self.T)

    def init_process(self):
        setup_seed(2024)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.gateway_id, world_size=self.size)

    def run(self):
        self.init_process()
        for r in range(self.rounds):
            self.download_global_model(self.server_id)
            for gr in range(self.gateway_rounds):
                sampled_clients, data_ratio = self.selection()  # select device
                self.borcast_model_and_info(sampled_clients)  # borcast global model and info
                self.aggregation_with_sample_clients(sampled_clients, data_ratio)  # aggregation gateway model
            self.info = extra_info(self.global_loss, self.gateway_id, self.T)
            gateway_weight_vec, info_len = pack(self.model, self.info)
            try:
                self.aggregation_with_cloud(gateway_weight_vec)# aggregation server model
            except:
                sys.exit()
            self.T+=1

    def aggregation_with_cloud(self,gateway_weight):
        load_weights(self.model,gateway_weight)
        gateway_weight_vec, info_len = pack(self.model, self.info)
        dist.send(gateway_weight_vec, self.server_id)
        # warning: don't recv form server

    def aggregation_with_sample_clients(self,sampled_clients,data_ratio):
        global_weight_vec, info_len = pack(self.model, self.info)
        weights_vec_list=[torch.zeros_like(global_weight_vec) for _ in range(len(sampled_clients))]
        for i,(c,s) in enumerate(zip(self.clients_list,sampled_clients)):
            if s==1:
                dist.recv(weights_vec_list[i], c)

        gateway_weight=torch.zeros_like(flatten_weights(extract_weights(self.model)))
        for i,weights_vec in enumerate(weights_vec_list):
            weights, info=unpack(weights_vec,info_len)
            gateway_weight+=weights*data_ratio[i]
            # global_weight+=weights/self.clients
        load_weights(self.model,gateway_weight)


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

    def borcast_model_and_info(self,sampled_clients):
        # borcast info
        sampled_clients=torch.tensor(sampled_clients)
        for i in self.clients_list:
            dist.send(sampled_clients,i)

        # borcast edge model
        global_weight_vec, info_len = pack(self.model, self.info)
        for i, c in zip(self.clients_list,sampled_clients):
            if c == 1: # be selected
                dist.send(global_weight_vec, i)