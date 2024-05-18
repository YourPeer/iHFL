import torch
import torch.nn as nn
import torch.nn.functional as F
from FLTask.data import data_distributer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class FedAvgNetMNIST(torch.nn.Module):
    def __init__(self, channel=1,input_dim=(16 * 4 * 4), hidden_dims=[120, 84], num_classes=10):
        super(FedAvgNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        # Replace F.relu with nn.ReLU modules
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x,get_features=False):
        features = {}
        x = self.pool(self.relu1(self.conv1(x)))

        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu3(self.fc1(x))
        z = self.relu4(self.fc2(x))
        x = self.fc3(z)

        if get_features:
            return x, z
        else:
            return x


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


class FedAvgNetTiny(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetTiny, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(16384, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu3(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


def get_topk_indices(tensor, k):
    _, indices = torch.topk(tensor.view(-1), k, largest=False)
    return set(indices.numpy())


def calculate_overlap_rate(model1, model2, k):
    overlap_count = 0
    total_count = 0
    params1 = (torch.abs(param) for name, param in model1.named_parameters() if 'weight' in name)
    params2 = (torch.abs(param) for name, param in model2.named_parameters() if 'weight' in name)
    for param1, param2 in zip(params1, params2):
        param_size=torch.numel(param1)
        topk_indices1 = get_topk_indices(param1, int(k*param_size))
        topk_indices2 = get_topk_indices(param2, int(k*param_size))
        overlap = topk_indices1.intersection(topk_indices2)
        print(len(overlap)/int(k*param_size))
        overlap_count += len(overlap)
        total_count += int(k*param_size)
    if total_count > 0:
        return overlap_count / total_count
    else:
        return 0


if __name__=="__main__":
    round=0
    model_path1="/home/archlab/lzr/NebulaCode/FLSaveModel/global_model_"+str(round)+"round.pth"
    model1=torch.load(model_path1)

    round = 99
    model_path2 = "/home/archlab/lzr/NebulaCode/FLSaveModel/global_model_" + str(round) + "round.pth"
    model2 = torch.load(model_path2)

    k = 0.2
    overlap_rate = calculate_overlap_rate(model1, model2, k)
    print(overlap_rate)
    #
    # # model=FedAvgNetCIFAR()
    distributer = data_distributer("../../../data","cifar10",64,10,'niid',0.6,2)
    activations1 = {}
    activations2 = {}


    def save_activation1(name):
        def hook(model, input, output):
            activations1[name] = output.detach().clone()
        return hook

    def save_activation2(name):
        def hook(model, input, output):
            activations2[name] = output.detach().clone()
        return hook


    for name, module in model1.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(save_activation1(name))

    for name, module in model2.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(save_activation2(name))

    indiceslist1=[]
    indiceslist2 = []
    data_iter=iter(distributer["local"][0]["train"])
    for i in range(20):
        x,_=next(data_iter)
        y1=model1(x)
        y2 = model2(x)

        # for layer_name, activation in activations.items():
        #     print(f"{layer_name}: {activation.shape}")

        activation1=activations1["relu2"].view(64,-1)
        activation2 = activations2["relu2"].view(64, -1)

        activation_tensor1 = activation1.gt(0).float()
        activation_tensor2 = activation2.gt(0).float()
        relu_activation1=torch.sum(activation_tensor1,dim=0)
        relu_activation2=torch.sum(activation_tensor2, dim=0)
        _, indices1 = torch.topk(relu_activation1, int(len(relu_activation1)*k), largest=False, dim=0)
        _, indices2 = torch.topk(relu_activation2, int(len(relu_activation2)*k), largest=False, dim=0)
        indiceslist1.append(indices1.numpy().tolist())
        indiceslist2.append(indices2.numpy().tolist())
    #
    def flatten(two_d_list):
        return [item for sublist in two_d_list for item in sublist]
    set_indices1 = set(flatten(indiceslist1))
    set_indices2 = set(flatten(indiceslist2))
    overlap_indices = set_indices1.intersection(set_indices2)
    print("Overlap of top-k smallest values:", len(overlap_indices)/int(len(relu_activation1)*k))



    # expanded1 = activation_tensor1.unsqueeze(1)
    # expanded2 = activation_tensor1.unsqueeze(0)
    #
    # expanded3 = activation_tensor2.unsqueeze(1)
    # expanded4 = activation_tensor2.unsqueeze(0)
    # hamming_distances1 = (expanded1 != expanded2).sum(dim=2).flatten()
    # hamming_distances2 = (expanded3 != expanded4).sum(dim=2).flatten()
    # _, indices1 = torch.topk(-hamming_distances1, 500, largest=True, dim=0)
    # _, indices2 = torch.topk(-hamming_distances2, 500, largest=True, dim=0)
    # set_indices1 = set(indices1.numpy())
    # set_indices2 = set(indices2.numpy())
    # overlap_indices = set_indices1.intersection(set_indices2)
    # print("Overlap of top-k smallest values:", len(overlap_indices))
    #
    # hamming_matrix = activation_tensor.shape[1]-hamming_distances
    # print(hamming_matrix)
    # normalized_matrix=(hamming_matrix-hamming_matrix.min())/(hamming_matrix.max()-hamming_matrix.min())
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(normalized_matrix, vmin=0, vmax=1,cmap='coolwarm')
    # plt.title("Heatmap of a Random Matrix")
    # plt.savefig("/home/archlab/lzr/NebulaCode/FLSaveModel/"+str(round)+".png")

