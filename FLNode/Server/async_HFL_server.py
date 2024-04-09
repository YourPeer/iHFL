from .sync_HFL_server import sync_HFL_server
import multiprocessing as mp
import torch.distributed as dist
from FLNode.tools import *
class async_HFL_server(sync_HFL_server):
    def __init__(self, c, args, distributer, model):
        super().__init__(c, args, distributer, model)
        self.T=0

    def run(self):
        self.init_process()
        lock = mp.Lock()
        self.borcast_model(self.topology.keys())
        for r in range(self.rounds):
            lock.acquire()
            self.async_aggregation()  # aggregation global model
            test_loss, test_acc = self.test_model()
            print(test_loss, test_acc)
            print("async agg")
            lock.release()

    def async_aggregation(self):
        info = extra_info(self.global_loss,self.server_id)
        global_weight_vec, info_len = pack(self.model, info)
        gateway_weight_vec=torch.zeros_like(global_weight_vec)
        dist.recv(gateway_weight_vec)
        global_weights, info = unpack(global_weight_vec, len(info))
        gateway_weights, (gateway_trainloss, gateway_id) = unpack(gateway_weight_vec, len(info))
        global_weights=0.7*global_weights+0.3*gateway_weights
        load_weights(self.model,global_weights)
        global_weight_vec, info_len = pack(self.model, info)
        dist.send(global_weight_vec, gateway_id)