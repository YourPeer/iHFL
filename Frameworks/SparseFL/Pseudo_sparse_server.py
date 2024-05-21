import copy
import os
import pickle
import random

import torch.distributed as dist
import torchvision.models
from torch import nn
import torch.nn.utils.prune as prune
from ..tools import *
import torch
import pandas as pd
from FLTools import flacos_pruner
class PSServer(object):
    def __init__(self, c, args, distributer, model):
        # System parameters
        self.server_id=c
        self.size=args.size
        self.clients = args.clients
        self.gpu_num = args.gpu_num
        self.port = args.port

        # Task parameters
        self.model = model
        self.test_loader = distributer["global"]["test"]
        self.rounds = args.rounds

        # Training parameters
        self.save_model_circle_round = 20
        self.global_loss=0.0
        data_map = distributer["data_map"]
        self.data_ratio = np.sum(data_map, axis=1) / np.sum(data_map)
        self.select_type=args.select_type
        self.select_ratio = args.select_ratio
        self.train_loss_list = torch.zeros(self.clients)

        # send extra info
        self.T=0
        self.info=extra_info(self.global_loss,self.server_id,self.T)

        # sparse parameters
        self.sparse = args.sparse
        self.fix = args.fix
        self.importanace_mode = args.importanace_mode
        self.pruning_ratio = args.pruning_ratio

        self.masks_list=None


    def init_process(self):
        setup_seed(2024)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.port)
        dist.init_process_group("gloo", rank=self.size-1, world_size=self.size)

    def run(self):
        self.init_process()
        logs = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Test Loss', 'Test Accuracy'])
        example_inputs, _ = next(iter(self.test_loader))
        for r in range(self.rounds):
            sampled_clients, data_ratio = self.selection() # select device
            sparse_models_list = self.sparse_models_for_clients(sampled_clients) # generate sub-model
            self.borcast_model_and_info(sampled_clients,sparse_models_list) # borcast global model and info
            self.aggregation_sparse_model(sampled_clients, data_ratio) # aggregation global model
            self.T+=1
            test_loss, test_acc=self.test_model()
            logs = logs.append(
                {'Epoch': r, 'Test Loss': test_loss, 'Test Loss': test_loss, 'Test Accuracy': test_acc},
                ignore_index=True)
            print(test_loss,test_acc)
            logs.to_csv('./FLRecordFile/sparseFL_record/niid0.6_unfixsnip_r0.05.csv', index=False)
            if r==0 or (r+1) % self.save_model_circle_round ==0:
                torch.save(self.model, './FLSaveModel/global_model_'+str(r)+'round.pth')

    def sparse_models_for_clients(self,sampled_clients):
        sparse_models_list = []
        if self.fix and self.masks_list:
            for i in sampled_clients: # use init mask
                sub_model = copy.deepcopy(self.model)
                self.apply_masks(sub_model, self.masks_list[i])
                sparse_models_list.append(sub_model)
        else:
            self.masks_list = []
            for i in range(self.clients):
                sub_model = copy.deepcopy(self.model)
                self.apply_pruning(sub_model, self.pruning_ratio)
                mask = self.extract_masks_as_vector(sub_model)
                self.remove_pruning(sub_model)
                sparse_models_list.append(sub_model)
                self.masks_list.append(mask)
        return sparse_models_list

    def apply_masks(self, model, masks):
        mask_index = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                num_weights = module.weight.numel()
                mask = masks[mask_index:mask_index + num_weights].view_as(module.weight)
                module.weight.data.mul_(mask)
                mask_index += num_weights

    def apply_pruning(self, model, prune_ratio=0.5):
        if self.importanace_mode == 'random':
            conv_prune_method=prune.random_structured
            fc_prune_method=prune.random_unstructured
        elif self.importanace_mode == 'weight':
            conv_prune_method=prune.l1_structured
            fc_prune_method = prune.l1_unstructured
        for name, module in model.named_modules():
            prune_ratio=prune_ratio*random.uniform(0.8,1)  # different ratio
            if isinstance(module, nn.Conv2d):
                conv_prune_method(module, name='weight', amount=prune_ratio, dim=0)
                if module.bias is not None:
                    fc_prune_method(module, name='bias', amount=prune_ratio)
            elif isinstance(module, nn.Linear):
                fc_prune_method(module, name='weight', amount=prune_ratio)
                if module.bias is not None:
                    fc_prune_method(module, name='bias', amount=prune_ratio)

    def extract_masks_as_vector(self, model):
        masks = []
        for module in model.modules():
            for attr in ['weight_mask', 'bias_mask']:
                if hasattr(module, attr):
                    mask = getattr(module, attr)
                    if mask is not None:
                        masks.append(mask.flatten())
        return torch.cat(masks)

    def remove_pruning(self, model):
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')
    def borcast_model_and_info(self,sampled_clients,sparse_models_list):
        # borcast info
        sampled_clients=torch.tensor(sampled_clients)
        for i in range(self.clients):
            dist.send(sampled_clients,i)
        # borcast globle model
        for i, c in enumerate(sampled_clients):
            if c == 1: # be selected
                global_weight_vec, info_len = pack(sparse_models_list[i], self.info)
                dist.send(global_weight_vec, i)
    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    def aggregation_sparse_model(self, sampled_clients, data_ratio):
        global_weight_vec, info_len = pack(self.model, self.info)
        weights_vec_list = [torch.zeros_like(global_weight_vec) for _ in range(len(sampled_clients))]
        for i, c in enumerate(sampled_clients):
            if c == 1:
                dist.recv(weights_vec_list[i], i)
        global_weight_num = self.get_num_parameters()
        global_weight = torch.zeros(global_weight_num)
        global_mask = torch.sum(torch.stack(self.masks_list), dim=0)
        for i, weights_vec in enumerate(weights_vec_list):
            sub_weight, info = unpack(weights_vec, len(self.info))
            global_weight += sub_weight  # *data_ratio[i]
        global_mask[global_mask == 0.0] = 1.0
        global_weight = global_weight / global_mask
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
        self.model.cpu()
        test_loss = test_loss / (batch_idx + 1)
        test_acc = 100. * correct / total
        return test_loss, test_acc

    def selection(self):
        if self.select_type=="random":
            sampled_clients = np.zeros(self.clients, dtype=int)
            selected_clients = np.random.choice(
                range(0, self.clients), int(self.clients*self.select_ratio), replace=False
            )
            selected_clients=sorted(selected_clients)
            sampled_clients[selected_clients]=1
            data_ratio = [ratio/np.sum(self.data_ratio[selected_clients]) for ratio in self.data_ratio]
        return sampled_clients,data_ratio