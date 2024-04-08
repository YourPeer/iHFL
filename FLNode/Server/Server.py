import os
import torch.distributed as dist
from ..tools import *
import torch
class Server(object):
    def __init__(self, args, distributer, model):
        # System parameters
        self.clients = args.clients
        self.gpu_num = args.gpu_num
        self.port = args.port

        # Task parameters
        self.model = model
        self.test_loader = distributer["global"]["test"]
        self.rounds = args.rounds

        # Training parameters
        self.global_loss=0.0
        data_map = distributer["data_map"]
        self.data_ratio = np.sum(data_map, axis=1) / np.sum(data_map)
        self.select_type=args.select_type
        self.select_ratio = args.select_ratio

    def init_process(self):
        setup_seed(2024)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.clients, world_size=self.clients+1)

    def run(self):
        self.init_process()
        for r in range(self.rounds):
            sampled_clients, data_ratio = self.selection() # select device
            self.borcast_model_and_info(sampled_clients) # borcast global model and info
            self.aggregation(sampled_clients,data_ratio) # aggregation global model
            test_loss, test_acc=self.test_model()
            print(test_loss,test_acc)

    def aggregation(self,sampled_clients,data_ratio):
        info=extra_info(self.global_loss)
        global_weight_vec, info_len = pack(self.model, info)
        weights_vec_list=[torch.zeros_like(global_weight_vec) for _ in range(len(sampled_clients))]
        for i,c in enumerate(sampled_clients):
            if c==1:
                dist.recv(weights_vec_list[i], i)

        global_weight=torch.zeros_like(flatten_weights(extract_weights(self.model)))
        for i,weights_vec in enumerate(weights_vec_list):
            weights, info=unpack(weights_vec,len(info))
            global_weight+=weights*data_ratio[i]
            # global_weight+=weights/self.clients
        load_weights(self.model, global_weight)

    def test_model(self):
        self.model.eval()
        self.model.cuda()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        criterion = torch.nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.model.train()
        self.model.cpu()
        test_loss = test_loss / (batch_idx + 1)
        test_acc = 100. * correct / total
        return test_loss, test_acc

    def selection(self):
        if self.select_type=="random":
            sampled_clients = np.zeros(self.clients, dtype=int)
            selected_clients = np.random.choice(
                range(0, self.clients - 1), int(self.clients*self.select_ratio), replace=False
            )
            selected_clients=sorted(selected_clients)
            sampled_clients[selected_clients]=1
            data_ratio = [ratio/np.sum(self.data_ratio[selected_clients]) for ratio in self.data_ratio]
        return sampled_clients,data_ratio

    def borcast_model_and_info(self,sampled_clients):
        # borcast info
        sampled_clients=torch.tensor(sampled_clients)
        for i in range(self.clients):
            dist.send(sampled_clients,i)

        # borcast globle model
        info = extra_info(self.global_loss)
        global_weight_vec, info_len = pack(self.model, info)
        for i, c in enumerate(sampled_clients):
            if c == 1: # be selected
                dist.send(global_weight_vec, i)
