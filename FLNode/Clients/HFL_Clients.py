from .Clients import Client
import torch.distributed as dist
from FLNode.tools import *
class HFL_Client(Client):
    def __init__(self, client_id, args, distributer, model):
        super().__init__(client_id, args, distributer, model)
        self.topology=args.topology
        self.gateway_id=[k for k,v in self.topology.items() if self.client_id in v][0]
        self.clients_list=self.topology[self.gateway_id]

    def run(self):
        self.init_process()
        for round in range(self.rounds):
            sampled_clients=self.download_info(len(self.clients_list), self.gateway_id) # selected device
            if sampled_clients[self.topology[self.gateway_id].index(self.client_id)]==1: # selected device perform training
                self.download_global_model(self.gateway_id)
                self.local_train()
                self.scheduler.step()
                self.info = extra_info(self.train_loss, self.client_id, self.T)
                self.aggregation()
            self.T+=1

    def aggregation(self):
        weights_vec, info_len=pack(self.model, self.info)
        dist.send(weights_vec, self.gateway_id)
