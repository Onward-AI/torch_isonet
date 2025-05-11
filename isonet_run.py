"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
"""


import torch
from ljp.engine import Trainer
from ljp.optimizers import Adan, AdaBound, DiffGrad, AdaBelief, CAME, Sophia
from ljp.dataset.hit import HIT
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from ljp.models.isonet import ISONet

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = f'cuda:0'
    layer_cfg = '4l064'
    kernel_size = 9
    train_bs = 32  # 8,16,32,64,128
    data = HIT(data_root='./hit_data',
               train_bs=train_bs,
               train_frac=0.1)

    model = ISONet(
        layer_cfg=layer_cfg,
        kernel_size=kernel_size,
        n_output=data.num_classes,
        inchannels=data.inchannels,
        activator='mish',
    )

    net = Trainer(model)

    net.compile(loss_dict={'loss_func': torch.nn.CrossEntropyLoss},
                optimizer_dict={'optimizer': Adan,
                                'lr': 5e-4,
                                # 'weight_decay': 1e-4
                                },
                lr_scheduler_dict={
                    'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                    'step_size': 60,
                    'gamma': 0.1,
                },
                metrics_dict={
                    "acc": Accuracy('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "precision": Precision('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "recall": Recall('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "f1": F1Score('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "auc": AUROC('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                },
                numpy_metric=False,
                monitor='val_acc',
                monitor_mode='max')
    net.fit(data=data, epochs=200, device=device, verbose=2, use_compile=False, use_amp=False,
            save_dir=f'/works/xiang/test/')
