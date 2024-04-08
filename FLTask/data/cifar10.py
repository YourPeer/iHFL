import torch.utils.data as data
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np

class CIFAR10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=False, transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 10

        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        base_dataset = CIFAR10(
            self.root, self.train, self.transform, None, self.download
        )

        data = base_dataset.data
        targets = np.array(base_dataset.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )

    return train_transform, valid_transform

def get_dataloader_cifar10(root, train=True, batch_size=32, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar10()

    if train:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )
        dataset.targets

    else:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=False, transform=valid_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader

def get_all_targets_cifar10(root, train=True):
    dataset = CIFAR10_truncated(root=root, train=train)
    all_targets = dataset.targets
    return all_targets

