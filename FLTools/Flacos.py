import copy

import torch
import torch.nn as nn
import torch_pruning as tp
from torchvision.models import vgg11,resnet50
class MyRandomImportance(tp.importance.Importance):
    def __init__(self):
        super(MyRandomImportance, self).__init__()
        self.group_imp_list = []
        #torch.manual_seed(42)

    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if isinstance(layer, nn.Conv2d) and prune_fn == tp.prune_conv_out_channels:
                local_norm = torch.rand(len(idxs))
                group_imp.append(local_norm)

            elif isinstance(layer, nn.Linear) and prune_fn == tp.prune_linear_out_channels:
                local_norm = torch.rand(len(idxs))
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

class MyMagnitudeImportance(tp.importance.Importance):
    def __init__(self):
        super(MyRandomImportance, self).__init__()
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
                self.imp = MyRandomImportance()
            elif self.importance_mode == "weight":
                self.imp = MyMagnitudeImportance()
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


def restore_model(full_model, pruned_model, mask_list, feature_output):

    mask_index = 0
    indices_old = None
    fc_indices_old=None
    for name, module in full_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, nn.Linear) and module.out_features == feature_output:
                continue

            pruned_module = dict(pruned_model.named_modules())[name]

            mask = mask_list[mask_index].bool()

            indices = torch.where(mask)[0]
            indices, _ = torch.sort(indices)
            print(indices)

            with torch.no_grad():
                if isinstance(module, nn.Conv2d):
                    new_weight = torch.zeros_like(module.weight)
                    #new_bias = torch.zeros_like(module.bias)
                    if indices_old is None:
                        new_weight[indices,:, :, :] = pruned_module.weight[:, :, :, :]
                        #new_bias[indices] = pruned_module.bias
                    else:
                        j=0
                        for i in range(indices.size(0)):
                            new_weight[indices[i], indices_old, :, :] = pruned_module.weight[j, :, :, :]
                            #new_bias[indices[i]] = pruned_module.bias[j]
                            j+=1
                        # new_weight[indices, indices_old, :, :] = pruned_module.weight[:, :, :, :]
                    module.weight.copy_(new_weight)
                    #module.bias.copy_(new_bias)
                    indices_old = torch.clone(indices)

                elif isinstance(module, nn.Linear):

                    new_weight = torch.zeros_like(module.weight)

                    if fc_indices_old is None:
                        m, n = pruned_module.weight.size(0), new_weight.size(1)
                        s = m // indices_old.size(0) # jiangge
                        for i, j in enumerate(indices_old):
                            new_weight[indices, j * s:(j + 1) * s] = pruned_module.weight[:, i * s:(i + 1) * s]
                    else:
                        j = 0
                        for i in range(indices.size(0)):
                            new_weight[indices[i], fc_indices_old] = pruned_module.weight[j, :]
                            j += 1
                        # new_weight[indices, indices_old, :, :] = pruned_module.weight[:, :, :, :]
                    module.weight.copy_(new_weight)
                    fc_indices_old = torch.clone(indices)

            mask_index += 1

#
# model = vgg11(pretrained=False,num_classes=10)
# full_model = copy.deepcopy(model)
# for param in full_model.parameters():
#     print(param.data)
#
# print(model)
# example_inputs = torch.randn(1, 3, 32, 32)
# pruner=flacos_pruner(model,0.5,example_inputs,iterative_steps=1,feature_output=10,importanace_mode="weight")
# pruner.prune(0.5)
# print(model)
# mask_list=pruner.get_remain_channel_index()[::-1]
# i=0
#
#
#
# feature_output = 10
# restore_model(full_model, model, mask_list, feature_output)
# for param in full_model.parameters():
#     print(param.data)
# 验证恢复后的模型
# i = 0
# for name, module in full_model.named_modules():
#     if isinstance(module, (nn.Conv2d, nn.Linear)) and not (isinstance(module, nn.Linear) and module.out_features == feature_output):
#         print(f"Layer {name} - {module}")
#         print(f"Parameter Size: {module.weight}")
#         i += 1


# print(model)
# # X=torch.rand(1,3,32,32)
# # y=model(X)
#
# model = vgg11(pretrained=False)
# example_inputs = torch.randn(1, 3, 32, 32)
# imp=clientMagnitudeImportance(mask)
# ignored_layers = []
# for m in model.modules():
#     if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
#         ignored_layers.append(m)  # DO NOT prune the final classifier!
#
# pruner = tp.pruner.MetaPruner(
#     model,
#     example_inputs,
#     importance=imp,
#     iterative_steps=1,
#     pruning_ratio=0.5,
#     pruning_ratio_dict=None,
#     ignored_layers=ignored_layers,
# )
# pruner.step()
# print(model)
