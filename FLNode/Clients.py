import torch
import torch.distributed as dist
import os
from .Comm_tools import *
class Client(object):
    def __init__(self, client_id, args, distributer, model):
        # System parameters
        self.clients=args.clients
        self.client_id=client_id
        self.gpu_num=args.gpu_num
        self.port=args.port

        # Task parameters
        self.model=model
        self.train_loader=distributer["local"][self.client_id]["train"]
        self.test_loader=distributer["global"]["test"]
        self.rounds=args.rounds
        self.local_steps=args.local_steps

        # Training parameters
        self.current_steps=0
        self.train_loss=0.0
        self.lr=args.learning_rate
        self.batchsize=args.batchsize
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[int(self.rounds * 0.6), int(self.rounds * 0.9)],gamma=0.1)

    def init_process(self):
        setup_seed(2024)
        torch.cuda.set_device(self.client_id % self.gpu_num)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.client_id, world_size=self.clients+1)

    def run(self):
        self.init_process()
        for round in range(self.rounds):
            self.local_train()
            self.scheduler.step()
            self.aggregation()

    def aggregation(self, server_id=-1):
        info=extra_info(self.train_loss)
        weights_vec, info_len=pack(self.model, info)
        if server_id==-1:
            server_id=self.clients
        dist.send(weights_vec, server_id)
        dist.recv(weights_vec, server_id)
        weights,info = unpack(weights_vec,info_len)
        load_weights(self.model,weights)

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