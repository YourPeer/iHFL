import copy
import os
import pickle

import torch.distributed as dist
import torchvision.models
from torch import nn

from ..tools import *
import torch
import pandas as pd
from FLTools import flacos_pruner
class Sp_server(object):
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

        # sparse parameters
        self.sparse=args.sparse
        self.fix = args.fix
        self.importanace_mode = args.importanace_mode
        self.pruning_ratio = args.pruning_ratio

        # send extra info
        self.T=0
        self.info=extra_info(self.global_loss,self.server_id,self.T)


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
            if self.sparse:
                sparse_models_list,masks_list=self.sparse_models_for_clients(example_inputs) # **************
                self.borcast_sparse_model_and_info(sparse_models_list,masks_list,sampled_clients) # borcast global model and info
                self.aggregation_sparse_model(sampled_clients, data_ratio, masks_list)  # aggregation global model
            else:
                self.borcast_model_and_info(sampled_clients) # borcast global model and info
                self.aggregation(sampled_clients,data_ratio) # aggregation global model
            self.T+=1
            test_loss, test_acc=self.test_model()
            logs = logs.append(
                {'Epoch': r, 'Test Loss': test_loss, 'Test Loss': test_loss, 'Test Accuracy': test_acc},
                ignore_index=True)
            print(test_loss,test_acc)
            logs.to_csv('./FLRecordFile/sparseFL_record/niid0.6_unfixsnip_r0.05.csv', index=False)
            if r==0 or (r+1) % self.save_model_circle_round ==0:
                torch.save(self.model, './FLSaveModel/global_model_'+str(r)+'round.pth')

    def borcast_sparse_model_and_info(self,sparse_models_list,masks_list,sampled_clients):
        sampled_clients = torch.tensor(sampled_clients)
        for i in range(self.clients):
            dist.send(sampled_clients, i)
        # Broadcast global model and info
        for i, c in enumerate(sampled_clients):
            if c == 1:
                sub_model = sparse_models_list[i]
                sub_mask=masks_list[i]
                model_data = {
                    'model_state_dict': sub_model,
                    # 'model_prune_mask': sub_mask,
                }
                buffer = pickle.dumps(model_data)
                buffer_tensor = torch.ByteTensor(list(buffer))
                dist.send(torch.tensor(len(buffer)), i)  # Send buffer size first
                dist.send(buffer_tensor, i)  # Then send buffer data

    def borcast_model_and_info(self,sampled_clients):
        # borcast info
        sampled_clients=torch.tensor(sampled_clients)
        for i in range(self.clients):
            dist.send(sampled_clients,i)

        # borcast globle model

        global_weight_vec, info_len = pack(self.model, self.info)
        for i, c in enumerate(sampled_clients):
            if c == 1: # be selected
                dist.send(global_weight_vec, i)
    def aggregation(self,sampled_clients,data_ratio):
        global_weight_vec, info_len = pack(self.model, self.info)
        weights_vec_list=[torch.zeros_like(global_weight_vec) for _ in range(len(sampled_clients))]
        for i,c in enumerate(sampled_clients):
            if c==1:
                dist.recv(weights_vec_list[i], i)
        global_weight=torch.zeros_like(flatten_weights(extract_weights(self.model)))

        for i,weights_vec in enumerate(weights_vec_list):
            weights, info=unpack(weights_vec,len(self.info))
            self.train_loss_list[i]=info[0].item() if info[0].item()!=0.0 else self.train_loss_list[i]
            # global_weight+=weights*data_ratio[i]
            global_weight+=weights/self.clients
        print(global_weight)
        load_weights(self.model, global_weight)

    def aggregation_sparse_model(self,sampled_clients,data_ratio,mask_list):
        sub_model_list=[]
        for i,c in enumerate(sampled_clients):
            if c==1:
                buffer_size = torch.tensor(0)
                dist.recv(buffer_size, src=i)  # Receive buffer size first
                buffer_tensor = torch.ByteTensor(buffer_size.item())
                dist.recv(buffer_tensor, src=i)  # Then receive buffer data
                model_data = pickle.loads(bytes(buffer_tensor.tolist()))
                sub_model = model_data['model_state_dict']
                sub_model_list.append(sub_model)


        global_weight=torch.zeros_like(flatten_weights(extract_weights(self.model)))
        generated_mask=torch.zeros_like(flatten_weights(extract_weights(self.model)))
        self_model_weight=flatten_weights(extract_weights(self.model))
        # print("global:"+str(self_model_weight))
        for i, (sub_model,mask)  in enumerate(zip(sub_model_list,mask_list)):
            full_model = copy.deepcopy(self.model)
            load_weights(full_model, torch.zeros_like(flatten_weights(extract_weights(self.model))))
            # if i==0:
            #     sub_model_weight = flatten_weights(extract_weights(sub_model))
            #     print("client:" + str(sub_model_weight))

            sub_model_restore=self.restore_model(full_model,sub_model,mask,10)
            model_restore_weight=flatten_weights(extract_weights(sub_model_restore))
            generated_mask+=(model_restore_weight != 0.0).float()
            # print("client_restore:"+str(model_restore_weight))
            # generated_mask += (model_restore_weight != 0.0).float()
            global_weight +=model_restore_weight#*data_ratio[i]

        generated_mask_copy=copy.deepcopy(generated_mask)

        generated_mask[generated_mask == 0.0]=1.0
        global_weight= global_weight/generated_mask

        variance_mask = (generated_mask_copy == 0.0).float()
        variance_weight=self_model_weight*variance_mask
        global_weight+=variance_weight
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
                range(0, self.clients), int(self.clients*self.select_ratio), replace=False
            )
            selected_clients=sorted(selected_clients)
            sampled_clients[selected_clients]=1
            data_ratio = [ratio/np.sum(self.data_ratio[selected_clients]) for ratio in self.data_ratio]
        return sampled_clients,data_ratio

    def restore_model(self,full_model, pruned_model, mask_list, feature_output):
        mask_index = 0
        indices_old = None
        fc_indices_old = None
        for name, module in full_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                pruned_module = dict(pruned_model.named_modules())[name]
                if isinstance(module, nn.Linear) and module.out_features == feature_output:
                    j = 0
                    new_weight = torch.zeros_like(module.weight)
                    for i in range(feature_output):
                        new_weight[i, fc_indices_old] = pruned_module.weight[j, :]
                        j += 1
                    with torch.no_grad():
                        module.weight.copy_(new_weight)
                    continue

                mask = mask_list[mask_index].bool()
                indices = torch.where(mask)[0]
                indices, _ = torch.sort(indices)
                with torch.no_grad():
                    if isinstance(module, nn.Conv2d):
                        new_weight = torch.zeros_like(module.weight)
                        if indices_old is None:
                            new_weight[indices, :, :, :] = pruned_module.weight[:, :, :, :]
                        else:
                            j = 0
                            for i in range(indices.size(0)):
                                new_weight[indices[i], indices_old, :, :] = pruned_module.weight[j, :, :, :]
                                j += 1
                            # new_weight[indices, indices_old, :, :] = pruned_module.weight[:, :, :, :]
                        module.weight.copy_(new_weight)
                        indices_old = torch.clone(indices)
                        mask_len=mask.size()[0]



                    elif isinstance(module, nn.Linear):
                        new_weight = torch.zeros_like(module.weight)
                        if fc_indices_old is None:
                            m,n  = pruned_module.weight.size(0), new_weight.size(1)  # 256,4096
                            s = n // mask_len #indices_old.size(0)  # jiange
                            for i, j in enumerate(indices_old): #32
                                j=j.item()
                                new_weight[indices, j * s : (j+1) * s] = pruned_module.weight[:, i * s:(i + 1) * s]
                        else:
                            j = 0
                            for i in range(indices.size(0)):
                                new_weight[indices[i], fc_indices_old] = pruned_module.weight[j, :]
                                j += 1
                            # new_weight[indices, indices_old, :, :] = pruned_module.weight[:, :, :, :]
                        module.weight.copy_(new_weight)
                        fc_indices_old = torch.clone(indices)

                mask_index += 1
        return full_model

    def sparse_models_for_clients(self,example_inputs):
        sparse_models_list = []
        masks_list = []
        for i in range(self.clients):
            sub_model = copy.deepcopy(self.model)
            pruner = flacos_pruner(sub_model, self.pruning_ratio, example_inputs, iterative_steps=1,
                                   feature_output=10,
                                   importanace_mode=self.importanace_mode)
            pruner.prune(self.pruning_ratio,self.fix)
            mask = pruner.get_remain_channel_index()
            sparse_models_list.append(sub_model)
            masks_list.append(mask)
        return sparse_models_list,masks_list



