from ..StandardFL.Clients import Client
import torch.distributed as dist
from ..tools import *
import pickle
class HFL_Client(Client):
    def __init__(self, client_id, args, distributer, model):
        super().__init__(client_id, args, distributer, model)
        self.client_id=client_id

    def run(self):
        self.init_process()
        self.topology = self.get_topology()
        self.gateway_id = [k for k, v in self.topology.items() if self.client_id in v][0]
        self.clients_list = self.topology[self.gateway_id]
        while 1:
            try:
                sampled_clients=self.download_info(len(self.clients_list), self.gateway_id) # selected device
            except:
                break
            if sampled_clients[self.topology[self.gateway_id].numpy().tolist().index(self.client_id)]==1: # selected device perform training
                self.download_global_model(self.gateway_id)
                self.local_train()
                self.scheduler.step()
                self.info = extra_info(self.train_loss, self.client_id, self.T)
                self.aggregation()
            self.T+=1

    def aggregation(self):
        weights_vec, info_len=pack(self.model, self.info)
        dist.send(weights_vec, self.gateway_id)

    def get_topology(self):
        buffer_tensor_size = torch.zeros(1, dtype=torch.long)
        dist.recv(buffer_tensor_size)
        buffer = torch.zeros(buffer_tensor_size[0], dtype=torch.uint8)
        dist.recv(buffer)
        topology = pickle.loads(buffer.numpy().tobytes())
        return topology
