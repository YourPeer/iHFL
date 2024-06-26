import pickle

from ..StandardFL.Server import Server
import torch.distributed as dist
from ..tools import *
class sync_HFL_server(Server):
    def __init__(self, c, args, distributer, model):
        super().__init__(c,args, distributer, model)
        self.args=args
        self.server_id=c


    def run(self):
        self.init_process()
        self.topology = self.generate_topology(self.args.clients, self.args.gateways)
        for r in range(self.rounds):
            self.borcast_model(self.topology.keys())
            self.aggregation()  # aggregation global model
            test_loss, test_acc = self.test_model()
            self.T+=1
            print(test_loss, test_acc)

    def aggregation(self):
        global_weight_vec, info_len = pack(self.model, self.info)
        weights_vec_list = [torch.zeros_like(global_weight_vec) for _ in range(len(self.topology))]
        for i,c in enumerate(self.topology.keys()):
            dist.recv(weights_vec_list[i], c)
        global_weight = torch.zeros_like(flatten_weights(extract_weights(self.model)))
        for i, weights_vec in enumerate(weights_vec_list):
            weights, info = unpack(weights_vec, info_len)
            global_weight += weights/len(self.topology)
        load_weights(self.model, global_weight)

    def borcast_model(self,gateways):
        # borcast globle model
        global_weight_vec, info_len = pack(self.model, self.info)
        for gateway in gateways:
            dist.send(global_weight_vec, gateway)

    def generate_topology(self, clients, gateways):
        topology_dict={i+clients:list(range(i, clients+gateways, gateways))[:-1] for i in range(gateways)}
        for key in topology_dict:
            topology_dict[key] = torch.tensor(topology_dict[key])
        serialized_data = pickle.dumps(topology_dict)
        data_tensor = torch.ByteTensor(list(serialized_data))
        data_tensor_size=torch.tensor([len(data_tensor)])
        for i in topology_dict.keys():
            dist.send(data_tensor_size,i)
        for i in topology_dict.keys():
            dist.send(data_tensor, i)
        return topology_dict