from .sync_HFL_server import sync_HFL_server
import multiprocessing as mp
import torch.distributed as dist
from FLNode.tools import *
class async_HFL_server(sync_HFL_server):
    def __init__(self, c, args, distributer, model):
        super().__init__(c, args, distributer, model)
        self.alpha=args.async_alpha
        self.staleness_func=args.staleness_func

    def run(self):
        self.init_process()
        lock = mp.Lock()
        self.borcast_model(self.topology.keys())
        for r in range(self.rounds):
            lock.acquire()
            self.async_aggregation()  # aggregation global model
            test_loss, test_acc = self.test_model()
            self.T+=1
            print(test_loss, test_acc)
            lock.release()

    def async_aggregation(self):
        global_weight_vec, info_len = pack(self.model, self.info)
        gateway_weight_vec=torch.zeros_like(global_weight_vec)
        dist.recv(gateway_weight_vec)
        global_weights, _ = unpack(global_weight_vec, info_len)
        gateway_weights, (gateway_trainloss, gateway_id, gateway_tao) = unpack(gateway_weight_vec, info_len)
        global_weights=self.staleness_aggregation(global_weights,gateway_weights,gateway_tao)
        load_weights(self.model,global_weights)
        global_weight_vec, info_len = pack(self.model, self.info)
        dist.send(global_weight_vec, gateway_id)

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