from .Server import Server
import torch.distributed as dist
from FLNode.tools import *
class sync_HFL_server(Server):
    def __init__(self, c, args, distributer, model):
        super().__init__(args, distributer, model)
        self.server_id=c
        self.topology=args.topology

    def run(self):
        self.init_process()
        for r in range(self.rounds):
            self.borcast_model(self.topology.keys())
            self.aggregation()  # aggregation global model
            test_loss, test_acc = self.test_model()
            print(test_loss, test_acc)

    def aggregation(self):
        info = extra_info(self.global_loss,self.server_id)
        global_weight_vec, info_len = pack(self.model, info)
        weights_vec_list = [torch.zeros_like(global_weight_vec) for _ in range(len(self.topology))]
        for i,c in enumerate(self.topology.keys()):
            dist.recv(weights_vec_list[i], c)
        global_weight = torch.zeros_like(flatten_weights(extract_weights(self.model)))
        for i, weights_vec in enumerate(weights_vec_list):
            weights, info = unpack(weights_vec, len(info))
            global_weight += weights/len(self.topology)
        load_weights(self.model, global_weight)

    def borcast_model(self,gateways):
        # borcast globle model
        info = extra_info(self.global_loss,self.server_id)
        global_weight_vec, info_len = pack(self.model, info)
        for gateway in gateways:
            dist.send(global_weight_vec, gateway)