import pickle

from ..StandardFL.Clients import Client
import torch.distributed as dist
from ..tools import *
def find_sublist_containing_number(data, number):
    for key, list_of_lists in data.items():
        for sublist in list_of_lists:
            if number in sublist:
                return sublist
    return None

class nebula_client(Client):
    def __init__(self, client_id, args, distributer, model):
        super().__init__(client_id, args, distributer, model)
        self.client_id=client_id



    def run(self):
        self.init_process()
        self.topology = self.get_topology()
        self.gateway_id, self.leader_id = [(k, i[0]) for k, v in self.topology.items() for i in v if self.client_id in i][0]
        self.clients_list = find_sublist_containing_number(self.topology, self.client_id)
        while 1:
            self.local_train()
            self.scheduler.step()
            self.info = extra_info(self.train_loss, self.client_id, self.T)
            if self.client_id != self.leader_id:
                self.aggregation_with_leader()
            else:
                self.aggregation_with_gateway()
                self.download_gateway_model()
            self.T+=1


    def aggregation_with_leader(self):
        weights_vec, info_len=pack(self.model, self.info)
        dist.send(weights_vec, self.leader_id)
        dist.recv(weights_vec, self.leader_id)
        load_weights(self.model, weights_vec)

    def aggregation_with_gateway(self):
        leader_weight_vec, info_len = pack(self.model, self.info)
        leader_weight, info = unpack(leader_weight_vec, info_len)
        weights_vec_list = [torch.zeros_like(leader_weight_vec) for _ in self.clients_list[1:]]
        for i,c in enumerate(self.clients_list[1:]):
            dist.recv(weights_vec_list[i],c)
            weights, info = unpack(weights_vec_list[i], info_len)
            leader_weight+=weights
        leader_weight/=len(self.clients_list)
        load_weights(self.model,leader_weight)
        weights_vec, info_len=pack(self.model, self.info)
        dist.send(weights_vec, self.gateway_id)


    def download_gateway_model(self):
        weights_vec, info_len=pack(self.model, self.info)
        dist.recv(weights_vec, self.gateway_id)
        for i in self.clients_list[1:]:
            dist.send(weights_vec, i)

    def get_topology(self):
        buffer_tensor_size = torch.zeros(1, dtype=torch.long)
        dist.recv(buffer_tensor_size)
        buffer = torch.zeros(buffer_tensor_size[0], dtype=torch.uint8)
        dist.recv(buffer)
        topology = pickle.loads(buffer.numpy().tobytes())
        return topology