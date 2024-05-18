import argparse
import torch
from FLTask import generated_task
from FLTools import Masking,CosineDecay
import pandas as pd
def add_sparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse', type=bool,default=True,help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', type=bool,default=False, help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization') #uniform/uniform_plus/ERK/ERK_plus/ER/snip/GraSP
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate used for dynamic sparse training (not used in this paper).')
    parser.add_argument('--density', type=float, default=0.2, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    args = parser.parse_args()
    return args

def get_args():
    parser = argparse.ArgumentParser()
    # System parameters
    parser.add_argument('--clients', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--port', type=int, default=2024)

    # Task parameters
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--model_name', type=str, default='cifarcnn')
    parser.add_argument('--data_type', type=str, default='iid')  # iid,niid,sharding_max
    parser.add_argument('--partition_dir', type=float, default=0.6)
    parser.add_argument('--partition_shards', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--local_steps', type=int, default=100)

    # Training  parameters
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--select_type', type=str, default="random")
    parser.add_argument('--select_ratio', type=float, default=0.5)
    parser.add_argument('--async_alpha', type=float, default=0.6)

    args = parser.parse_args()
    return args

def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    criterion = torch.nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def local_train(model, optimizer, criterion,round,train_loader,test_loader, mask,logs):
    model.train()
    train_loss=0.0
    for r in range(round):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            data, targets= inputs.cuda(non_blocking=True),targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            # forward pass
            data, targets = data.cuda(), targets.cuda()
            output = model(data)
            loss = criterion(output, targets)
            train_loss+=loss.item()
            # backward pass
            loss.backward()
            if mask is not None:
                mask.step()
            else:
                optimizer.step()
        train_loss/=batch_idx
        test_loss,test_acc=test_model(model, test_loader)
        print(train_loss,test_loss,test_acc)
        logs = logs.append({'Epoch': r, 'Train Loss': train_loss, 'Test Loss': test_loss, 'Test Accuracy': test_acc}, ignore_index=True)
        logs.to_csv('./FLRecordFile/sparse_record/uniform_unfix.csv', index=False)

if __name__=="__main__":
    sparse_args=add_sparse_args()
    args = get_args()
    distributer, model = generated_task(args.data_path, args.dataset_name, args.model_name, args.clients,
                                        args.batchsize, args.data_type, args.partition_dir, args.partition_shards)
    model.cuda()
    train_loader = distributer["local"][0]["train"]
    test_loader = distributer["global"]["test"]
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    mask = None
    print(sparse_args.sparse)
    if sparse_args.sparse:
        decay = CosineDecay(sparse_args.death_rate, len(train_loader) * (args.rounds))
        mask = Masking(optimizer, death_rate=sparse_args.death_rate, death_mode=sparse_args.death, death_rate_decay=decay,
                       growth_mode=sparse_args.growth,
                       redistribution_mode=sparse_args.redistribution, args=sparse_args, train_loader=train_loader)
        mask.add_module(model, sparse_init=sparse_args.sparse_init, density=sparse_args.density)
    logs = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
    local_train(model, optimizer, criterion, args.rounds, train_loader,test_loader, mask,logs)
