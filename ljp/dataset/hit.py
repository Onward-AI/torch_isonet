"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com

        DATASET PAPER: Lei Hou, Haiming Yi, Yuhong Jin, Min Gui, Lianzheng Sui,
        Jianwei Zhang, and Yushu Chen. 2023. Inter-shaft Bearing Fault Diagnosis
        Based on Aero-engine System: A Benchmarking Dataset Study,
        Journal of Dynamics, Monitoring and Diagnostics, 2: 228-42.
        https://doi.org/10.37965/jdmd.2023.314
"""
import os
import torch
import numpy
import pandas
from torch.utils.data import DataLoader, TensorDataset


class HIT:


    def __init__(self,
                 data_root=f'path_to_your_HIT_dataset/',
                 train_bs=64,
                 train_frac=1.0,
                 shuffle=True):
        super().__init__()
        self.batch_size = train_bs
        self.shuffle = shuffle
        self.data_path = data_root
        self.input_shape = [self.batch_size, 1, 1, 2048]

        hit_train_ = numpy.load(os.path.join(data_root, 'train_x.npy'))
        labels_train_ = numpy.load(os.path.join(data_root, 'train_y.npy'))
        train = pandas.concat(
            [pandas.DataFrame(hit_train_), pandas.DataFrame(labels_train_, columns=['label'])], axis=1)
        health = train[train.label == 0]
        inner = train[train.label == 1]
        outer = train[train.label == 2]
        hit_train = pandas.concat(
            [
                health.sample(frac=train_frac, random_state=2024),
                inner.sample(frac=train_frac, random_state=2024),
                outer.sample(frac=train_frac, random_state=2024),
            ],
            axis=0)
        labels_train = hit_train.pop('label').to_numpy()
        hit_train = hit_train.to_numpy()
        hit_test = numpy.load(os.path.join(data_root, 'test_x.npy'))
        labels_test = numpy.load(os.path.join(data_root, 'test_y.npy'))
        print(f'Train: {hit_train.shape}')
        print(pandas.DataFrame(labels_train).value_counts())
        print(f'Test: {hit_test.shape}')
        print(pandas.DataFrame(labels_test).value_counts())

        hit_train = numpy.expand_dims(hit_train, 1)
        hit_train = numpy.expand_dims(hit_train, 1)
        hit_test = numpy.expand_dims(hit_test, 1)
        hit_test = numpy.expand_dims(hit_test, 1)
        x_torch_train = torch.from_numpy(hit_train).type(torch.FloatTensor)
        x_torch_test = torch.from_numpy(hit_test).type(torch.FloatTensor)
        y_torch_train = torch.from_numpy(labels_train.reshape(-1)).type(torch.long)
        y_torch_test = torch.from_numpy(labels_test.reshape(-1)).type(torch.long)
        td_train = TensorDataset(x_torch_train, y_torch_train)
        td_test = TensorDataset(x_torch_test, y_torch_test)
        if torch.cuda.is_available():
            pin_memory = True
        else:
            pin_memory = False
        self.testloader = DataLoader(dataset=td_test,
                                     batch_size=512,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=pin_memory)

        self.trainloader = DataLoader(dataset=td_train,
                                      batch_size=train_bs,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=pin_memory)
        self.desc = f'HIT[bz{train_bs}f{train_frac}]'
        self.num_classes = 3
        self.inchannels = 1
