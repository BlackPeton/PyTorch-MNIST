import torch
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])


def loadTrain_set(data_set_path):
    train_set = datasets.MNIST(
        root=data_set_path, transform=transform, train=True, download=True)
    return train_set


def loadTest_set(data_set_path):
    test_set = datasets.MNIST(
        root=data_set_path, transform=transform, train=False)
    return test_set
