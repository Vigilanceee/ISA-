"""
数据加载器 (支持 DDP DistributedSampler)

电压编码:
  忆阻器 (reram/pcm/stt):  [-0.5V, +0.5V]  mean=0.5, std=1.0
  晶体管 (fefet/flash):    [ 0.0V, +4.0V]  mean=0.0, std=0.25
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

_TRANSISTOR_DEVICES = {'fefet', 'flash'}


def get_cifar10_loaders(batch_size: int = 128,
                        data_dir: str = './data',
                        num_workers: int = 0,
                        distributed: bool = False,
                        device_type: str = 'reram'):
    """
    返回 (train_loader, val_loader[, train_sampler])。

    当 distributed=True 时额外返回 train_sampler，
    供训练循环在每个 epoch 开始时调用 sampler.set_epoch(epoch)。
    """
    if device_type in _TRANSISTOR_DEVICES:
        norm = transforms.Normalize((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    else:
        norm = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True,
                                 download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=data_dir, train=False,
                                 download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_set) if distributed else None
    val_sampler   = DistributedSampler(test_set, shuffle=False) if distributed else None

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            multiprocessing_context='spawn',
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        **worker_kwargs)
    val_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        **worker_kwargs)

    if distributed:
        return train_loader, val_loader, train_sampler
    return train_loader, val_loader


def get_mnist_loaders(batch_size: int = 128,
                      data_dir: str = './data',
                      num_workers: int = 0,
                      distributed: bool = False,
                      device_type: str = 'reram'):
    """
    返回 MNIST 的 (train_loader, val_loader[, train_sampler])。
    """
    if device_type in _TRANSISTOR_DEVICES:
        norm = transforms.Normalize((0.0,), (0.25,))
    else:
        norm = transforms.Normalize((0.5,), (1.0,))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    train_set = datasets.MNIST(root=data_dir, train=True,
                               download=True, transform=transform_train)
    test_set = datasets.MNIST(root=data_dir, train=False,
                              download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_set) if distributed else None
    val_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            multiprocessing_context='spawn',
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        **worker_kwargs)
    val_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        **worker_kwargs)

    if distributed:
        return train_loader, val_loader, train_sampler
    return train_loader, val_loader
