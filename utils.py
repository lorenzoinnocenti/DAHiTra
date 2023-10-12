import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.data import Sampler

import data_config
from datasets.CD_dataset import CDDataset, xBDataset, xBDatasetMulti


# def get_loader(data_name, img_size=256, batch_size=8, split='test',
#                is_train=False, dataset='CDDataset', patch=None):
#     dataConfig = data_config.DataConfig().get_data_config(data_name)
#     root_dir = dataConfig.root_dir
#     label_transform = dataConfig.label_transform
#     print(dataConfig)
# 
#     if dataset == 'CDDataset':
#         data_set = CDDataset(root_dir=root_dir, split=split,
#                                  img_size=img_size, is_train=is_train,
#                                  label_transform=label_transform, patch=patch)
#     elif dataset == 'xBDataset':
#         data_set = xBDataset(root_dir=root_dir, split=split,
#                                  img_size=img_size, is_train=is_train,
#                                  label_transform=label_transform)
#     elif dataset == 'xBDatasetMulti':
#         data_set = xBDatasetMulti(root_dir=root_dir, split=split,
#                                  img_size=img_size, is_train=is_train,
#                                  label_transform=label_transform)
#     else:
#         raise NotImplementedError(
#             'Wrong dataset name %s (choose one from [CDDataset])'
#             % dataset)
# 
#     shuffle = is_train
#     dataloader = DataLoader(data_set, batch_size=batch_size,
#                                  shuffle=False, num_workers=8)
# 
#     return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    elif args.dataset == 'xBDataset':
        training_set = xBDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = xBDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    elif args.dataset == 'xBDatasetMulti':
        training_set = xBDatasetMulti(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = xBDatasetMulti(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)
    nr_images_epoch = training_set.__len__() - training_set.__len__()%args.batch_size
    # nr_images_epoch = 7444
    sampler = RandomSampler(training_set, num_samples=nr_images_epoch) 
    train_dataloader = DataLoader(training_set, batch_size=args.batch_size,
                                 num_workers=args.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    # datasets = {'train': training_set, 'val': val_set}
    # dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
    #                              shuffle=True, num_workers=args.num_workers)
    #                for x in ['train', 'val']}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    # tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples