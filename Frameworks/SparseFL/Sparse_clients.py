import copy
import pickle
from FLTools import flacos_pruner
import torch
import torch_pruning as tp
import torch.distributed as dist
from ..tools import *
import os
from FLTools import Masking,CosineDecay
class Sp_client(object):
    def __init__(self, client_id, args, distributer, model):
        self.args=args
        # System parameters
        self.size = args.size
        self.clients=args.clients
        self.client_id=client_id
        self.server_id=self.clients
        self.gpu_num=args.gpu_num
        self.port=args.port

        # Task parameters
        self.model=model
        self.train_loader=distributer["local"][self.client_id]["train"]
        self.test_loader=distributer["global"]["test"]
        self.rounds=args.rounds
        self.local_steps=args.local_steps

        # Training parameters
        self.save_model_circle_round=20
        self.current_steps=0
        self.train_loss=0.0
        self.lr=args.learning_rate
        self.batchsize=args.batchsize
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[int(self.rounds * 0.6), int(self.rounds * 0.9)],gamma=0.1)

        # sparse parameters
        self.sparse = args.sparse
        self.fix = args.fix
        self.importanace_mode = args.importanace_mode
        self.pruning_ratio = args.pruning_ratio

        # send extra info
        self.T = 0
        self.info = extra_info(self.train_loss, self.client_id, self.T)



    def init_process(self):
        setup_seed(2024)
        torch.cuda.set_device(self.client_id % self.gpu_num)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.client_id, world_size=self.size)

    def run(self):
        self.init_process()
        # self.model.cuda()


        for round in range(self.rounds):
            sampled_clients=self.download_info(self.clients,self.server_id) # selected device
            if sampled_clients[self.client_id]==1: # selected device perform training
                if self.sparse:
                    self.download_submodel(self.server_id)
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
                    # if dist.get_rank()==0:
                    #     for param in self.model.parameters():
                    #         print(param.data)
                    self.local_train()
                    self.sparse_aggregation()
                else:
                    self.download_global_model(self.server_id)
                    self.local_train()
                    self.aggregation()

                self.info = extra_info(self.train_loss, self.client_id, self.T)
            self.T+=1
            if round==0 or (round+1) % self.save_model_circle_round ==0 and self.client_id in [0,1]:
                torch.save(self.model, './FLSaveModel/'+str(self.client_id)+'_local_model_'+str(round)+'_round.pth')

    def download_global_model(self,server_id):
        weights_vec, info_len = pack(self.model, self.info)
        dist.recv(weights_vec,server_id)
        weights, info = unpack(weights_vec, info_len)
        load_weights(self.model, weights)

    def download_info(self,clients,server_id):
        sampled_clients = torch.zeros(clients, dtype=int)
        dist.recv(sampled_clients,server_id)
        return sampled_clients

    def sparse_aggregation(self):
        model_data = {
            'model_state_dict': self.model,
            # 'model_prune_mask': sub_mask,
        }
        buffer = pickle.dumps(model_data)
        buffer_tensor = torch.ByteTensor(list(buffer))
        dist.send(torch.tensor(len(buffer)), self.server_id)  # Send buffer size first
        dist.send(buffer_tensor, self.server_id)  # Then send buffer data

    def aggregation(self):
        weights_vec, info_len = pack(self.model, self.info)
        dist.send(weights_vec, self.server_id)


    def local_train(self):
        self.model.cuda()
        self.model.train()
        self.train_loss=0.0
        for _ in range(self.local_steps):
            (data, targets) = self.get_batch_data()
            self.optimizer.zero_grad()
            # forward pass
            data, targets = data.cuda(), targets.cuda()
            output = self.model(data)
            loss = self.criterion(output, targets)
            self.train_loss+=loss.item()
            # backward pass
            loss.backward()
            self.optimizer.step()
        self.train_loss/=self.local_steps
        self.scheduler.step()
        self.model.cpu()


    def get_batch_data(self):
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self.train_loader)
            batch_data = next(self.data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.local_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def test_model(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        criterion = torch.nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.model.train()
        self.model.cpu()
        test_loss=test_loss / (batch_idx + 1)
        test_acc=100. * correct / total
        return test_loss,test_acc

    def download_submodel(self,server_id):
        buffer_size = torch.tensor(0)
        dist.recv(buffer_size, src=server_id)  # Receive buffer size first
        buffer_tensor = torch.ByteTensor(buffer_size.item())
        dist.recv(buffer_tensor, src=server_id)  # Then receive buffer data
        model_data = pickle.loads(bytes(buffer_tensor.tolist()))
        self.model = model_data['model_state_dict']

        # sub_mask = model_data['model_prune_mask']
        # example_inputs, _ = next(iter(self.test_loader))
        # pruner = flacos_pruner(self.model, self.pruning_ratio, example_inputs, iterative_steps=1, feature_output=1000,
        #                        importanace_mode="client")
        # pruner.prune(self.pruning_ratio,sub_mask)
        # self.model.load_state_dict(sub_model)



