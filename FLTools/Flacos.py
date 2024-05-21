import copy

import torch
import torch.nn as nn
import torch_pruning as tp


class FedAvgNetCIFAR(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2,bias=False)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2,bias=False)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512,bias=False)
        self.classifier = nn.Linear(512, num_classes,bias=False)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()



    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu1(x)

        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)

        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu3(self.linear_1(x))
        y = self.classifier(z)
        x = self.relu4(y)

        if get_features:
            return y, z

        else:
            return y


class RandomImportance(tp.importance.Importance):
    def __init__(self):
        super(RandomImportance, self).__init__()
        self.group_imp_list = []
        torch.manual_seed(42)

    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                local_norm = torch.rand(len(idxs))
                print(len(idxs))
                group_imp.append(local_norm)

            elif isinstance(layer, nn.Linear) and prune_fn == tp.prune_linear_out_channels:
                local_norm = torch.rand(len(idxs))
                print(len(idxs))
                group_imp.append(local_norm)

            elif isinstance(layer, nn.BatchNorm2d) and prune_fn == tp.prune_batchnorm_out_channels:
                local_norm = torch.rand(len(idxs))

                group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        self.group_imp_list.append(group_imp)
        return group_imp

    def get_group_imp(self):
        return self.group_imp_list

class MagnitudeImportance(tp.importance.Importance):
    def __init__(self):
        super(RandomImportance, self).__init__()
        self.group_imp_list = []
    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().sum(1)
                group_imp.append(local_norm)

            elif isinstance(layer, nn.Linear) and prune_fn == tp.prune_linear_out_channels:
                w = layer.weight.data[idxs]
                local_norm = w.abs().sum(1)
                group_imp.append(local_norm)

            elif isinstance(layer, nn.BatchNorm2d) and prune_fn == tp.prune_batchnorm_out_channels:
                w = layer.weight.data[idxs]
                local_norm = w.abs()
                group_imp.append(local_norm)


        if len(group_imp)==0:return None
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        self.group_imp_list.append(group_imp)
        return group_imp

    def get_group_imp(self):
        return self.group_imp_list


class clientMagnitudeImportance(tp.importance.Importance):
    def __init__(self, mask):
        super(clientMagnitudeImportance, self).__init__()
        self.mask=mask
        self.i = 0
    def __call__(self, group, **kwargs):
        group_imp =  self.mask[self.i]
        self.i+=1
        return group_imp

class flacos_pruner(object):
    def __init__(self, model,pruning_ratio, feature_input,iterative_steps=1,feature_output=1000,importanace_mode="random"):
        self.model=model
        self.feature_input=feature_input
        self.feature_output = feature_output
        self.importance_mode="random"
        self.iterative_steps=iterative_steps
        self.pruning_ratio=pruning_ratio
        self.group_imp_list=[]
        self.imp=None

    def prune(self,pruning_ratio=0.5, fix=True,mask=None):
        self.pruning_ratio=pruning_ratio
        if self.imp is None or not fix:
            if self.importance_mode == "random":
                self.imp = RandomImportance()
            elif self.importance_mode == "weight":
                self.imp = MagnitudeImportance()
            elif self.importance_mode == "client":
                self.imp = clientMagnitudeImportance(mask)
        self.group_imp_list=self.imp.get_group_imp()
        ignored_layers = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.feature_output:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

        pruner = tp.pruner.MetaPruner(
            self.model,
            self.feature_input,
            importance=self.imp,
            iterative_steps=self.iterative_steps,
            pruning_ratio=self.pruning_ratio,
            pruning_ratio_dict=None,
            ignored_layers=ignored_layers,
        )
        #base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        pruner.step()
        # macs, nparams = tp.utils.count_ops_and_params(self.model, self.feature_input)
        # print("  Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        # print("  MACs: %.2f G => %.2f G"% (base_macs / 1e9, macs / 1e9))

    def get_remain_channel_index(self):
        mask=[]
        for group_imp in self.group_imp_list:
            flat_tensor = group_imp.flatten()
            k = int(flat_tensor.numel() * (1-self.pruning_ratio))
            _, indices = torch.topk(flat_tensor, k)
            binary_tensor = torch.zeros_like(flat_tensor)
            binary_tensor[indices] = 1
            mask.append(binary_tensor)
        return mask[::-1]


# model=FedAvgNetCIFAR()
# print(model)
# example_inputs=torch.rand(1,3,32,32)
# pruner = flacos_pruner(model, 0.5, example_inputs, iterative_steps=1,
#                                    feature_output=10,
#                                    importanace_mode="random")
# pruner.prune()
# state_dict = tp.state_dict(model)
# for param in model.parameters():
#     print(param.size())
#     break
# print(state_dict['full_state_dict'])
# new_model=FedAvgNetCIFAR().eval()
# tp.load_state_dict(new_model, state_dict=state_dict)
# for param in new_model.parameters():
#     print(param.size())
#     break

