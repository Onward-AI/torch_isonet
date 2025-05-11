# -*- coding: utf-8 -*-
"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
"""
import functools
import os
import time
import datetime
import copy
import typing
import platform
import torch
import numpy
import pandas
from tqdm import tqdm

from ljp.logger import get_logger
from prettytable import PrettyTable


def get_parameter_quantity(net):
    return sum(p.numel() for p in net.parameters()), \
        sum(p.numel() for p in net.parameters() if p.requires_grad)


def print_bar(logger):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger("\n" + "=" * 120 + "%s" % nowtime + "=" * 120)


def get_describle(a: dict = {}, detail=True):
    desc = str()
    for it in a.items():
        if type(it[1]) == bool:
            if it[1]:
                s = str(it[0]).replace('_', '').replace(' ', '')
                desc += f"_{s}"
        else:
            s1 = str(it[0]).replace('_', '').replace(' ', '')
            s2 = str(it[1]).replace('_', '').replace(' ', '')
            desc += f"_{s1}{s2}"
    if detail:
        return desc
    else:
        return ''


def update_metrics_results(metrics_dict: dict,
                           trues: torch.Tensor,
                           preds: torch.Tensor,
                           metrics: dict = {},
                           prefix: str = '',
                           to_numpy: bool = True) -> None:
    trues, preds = [trues.cpu().numpy(), preds.cpu().numpy()] if to_numpy else [trues, preds]
    for name, metric_func in metrics_dict.items():
        r = metric_func(preds, trues)
        metrics[prefix + name] = metric_func(preds, trues).item() if isinstance(r, torch.Tensor) else r


def get_std_loss_and_metrics_results(
        net: torch.nn.Module,
        metrics_dict: dict,
        trues: list,
        preds: list,
        loss_func: torch.nn.Module,
        prefix: str,
        numpy_metric: bool = True
) -> typing.Tuple[dict, dict]:
    out, labels = torch.cat(preds, dim=0), torch.cat(trues, dim=0)
    epoch_metrics = {}
    epoch_metrics[prefix + "loss"] = loss_func(out, labels).item()
    update_metrics_results(metrics_dict, trues=labels, preds=out.detach(), metrics=epoch_metrics, prefix=prefix,
                           to_numpy=numpy_metric)
    return epoch_metrics, {'preds': out.detach().cpu().numpy(), 'trues': labels.cpu().numpy()}




# @torch.set_grad_enabled(True)
def forward_train(
        net: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        features: torch.Tensor,
        labels: torch.Tensor,
        retain_graph: bool = False,
) -> typing.Tuple[float, torch.Tensor]:
    if hasattr(optimizer, 'optimizer'):
        optimizer.optimizer.zero_grad()
    else:
        optimizer.zero_grad()
    out = net(features)
    loss = loss_func(out, labels)
    loss.backward(retain_graph=retain_graph)
    optimizer.step()
    return loss.item(), out


# @torch.set_grad_enabled(True)
def forward_train_amp(
        net: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        features: torch.Tensor,
        labels: torch.Tensor,
        scaler: torch.cuda.amp.GradScaler,
        max_norm: float = 1e-5,
        retain_graph: bool = False,
) -> typing.Tuple[float, torch.Tensor]:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = net(features)
        loss = loss_func(out, labels)
    scaler.scale(loss).backward(retain_graph=retain_graph)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), out


@torch.no_grad()
def forward_test(
        net: torch.nn.Module,
        loss_func: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
) -> typing.Tuple[float, torch.Tensor]:
    out = net(features)
    batch_loss = loss_func(out, labels).item()
    return batch_loss, out


def trainOneEpoch(
        net: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics_dict: dict,
        loader: torch.utils.data.DataLoader,
        forward_func: typing.Callable,
        metric_func: typing.Callable = get_std_loss_and_metrics_results,
        numpy_metric: bool = True,
        device: str = 'cpu',
        verbose: int = 1) -> typing.Tuple[dict, dict]:
    net.train()
    total, prefix, batch_metrics = 0, '', {}
    tqdm_iterator = tqdm(loader,
                         desc=f'Training @ {platform.node()} {device}',
                         leave=True,
                         ncols=250,
                         unit='batchs',
                         disable=(verbose != 2))
    trues, preds = [], []

    def inner_run(features, labels, total):
        if type(features) == list:
            features = [f.to(device) for f in features]
            true_labels = features[-1]
        else:
            features = features.to(device)
            true_labels = labels.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        batch_loss, out = forward_func(net=net,
                                       loss_func=loss_func,
                                       optimizer=optimizer,
                                       features=features,
                                       labels=labels,
                                       retain_graph=False)
        preds.append(out)
        trues.append(true_labels)
        if verbose == 2:
            batch_metrics[prefix + "loss"] = batch_loss
            # update_metrics_results(metrics_dict, trues=true_labels, preds=out.detach(), metrics=batch_metrics,
            #                        prefix=prefix, to_numpy=numpy_metric)
        tqdm_iterator.set_postfix({k: round(batch_metrics[k], 4) for k in batch_metrics.keys()})

    for data in tqdm_iterator:
        if len(data) == 1:
            features, labels = data[0]['data'], data[0]['label']
            labels = labels.squeeze(1).long()
        elif len(data) == 2:
            features, labels = data
        else:
            labels = data.pop()
            features = data
        inner_run(features, labels, total)

    return metric_func(
        net=net,
        metrics_dict=metrics_dict,
        trues=trues,
        preds=preds,
        loss_func=loss_func,
        prefix=prefix,
        numpy_metric=numpy_metric,
    )


def testOneEpoch(net: torch.nn.Module,
                 loss_func: torch.nn.Module,
                 metrics_dict: dict,
                 loader: torch.utils.data.DataLoader,
                 forward_func: typing.Callable,
                 metric_func: typing.Callable = get_std_loss_and_metrics_results,
                 numpy_metric: bool = True,
                 device: str = 'cpu',
                 verbose: int = 1) -> typing.Tuple[dict, dict]:
    net.eval()
    total, prefix, batch_metrics = 0, 'val_', {}
    tqdm_iterator = tqdm(loader,
                         desc=f'Testing @ {device}',
                         leave=True,
                         ncols=250,
                         unit='batchs',
                         disable=(verbose != 2))
    true, pred = [], []

    def inner_run(features, labels, total):
        if type(features) == list:
            features = [f.to(device) for f in features]
            true_labels = features[-1]
        else:
            features = features.to(device)
            true_labels = labels.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        batch_loss, out = forward_func(net=net, loss_func=loss_func, features=features, labels=labels)
        pred.append(out)
        true.append(true_labels)
        if verbose == 2:
            batch_metrics[prefix + "loss"] = batch_loss
            # update_metrics_results(metrics_dict, trues=true_labels, preds=out.detach(), metrics=batch_metrics,
            #                        prefix=prefix, to_numpy=numpy_metric)
        tqdm_iterator.set_postfix({k: round(batch_metrics[k], 4) for k in batch_metrics.keys()})

    for data in tqdm_iterator:
        if len(data) == 1:
            features, labels = data[0]['data'], data[0]['label']
            labels = labels.squeeze(1).long()
        elif len(data) == 2:
            features, labels = data
        else:
            labels = data.pop()
            features = data
        inner_run(features, labels, total)

    return metric_func(
        net=net,
        metrics_dict=metrics_dict,
        trues=true,
        preds=pred,
        loss_func=loss_func,
        prefix=prefix,
        numpy_metric=numpy_metric, )


class Engine:
    def __init__(self, net, desc=''):
        self.net = net
        self.desc = net.desc if hasattr(net, 'desc') else desc
        self.params, self.params_trainable = get_parameter_quantity(net)
        self.forward_func_train = None
        self.forward_func_train_amp = None
        self.forward_func_test = None
        self.metric_func = None
        self.trainOneEpoch = None
        self.testOneEpoch = None

    def compile(self,
                loss_dict=None,
                optimizer_dict=None,
                lr_scheduler_dict={},
                metrics_dict=None,
                numpy_metric=True,
                monitor='val_loss',
                monitor_mode='min'):
        assert monitor_mode in ['min', 'max']
        assert monitor in list(metrics_dict.keys()) + ['loss'] + \
               list(map(lambda x: 'val_' + x, list(metrics_dict.keys()) + [
                   'loss'])), f"{monitor} not in {list(metrics_dict.keys()) + ['loss'] + list(map(lambda x: 'val_' + x, list(metrics_dict.keys()) + ['loss']))}"
        if loss_dict and loss_dict.__contains__('loss_func'):
            loss_func = loss_dict['loss_func']
            loss_dict.pop('loss_func')
            self.loss_func = loss_func(**loss_dict)
            self.loss_desc = f'-{type(self.loss_func).__name__}{get_describle(loss_dict)}'
        else:
            self.loss_func = torch.nn.MSELoss()
            self.loss_desc = f'-{type(self.loss_func).__name__}'
        if optimizer_dict and optimizer_dict.__contains__('optimizer'):
            optimizer = optimizer_dict['optimizer']
            optimizer_dict.pop('optimizer')
            enhancer, enhancer_dict = None, None
            if optimizer_dict.__contains__('enhancer') and optimizer_dict.__contains__('enhancer_dict'):
                enhancer = optimizer_dict['enhancer']
                enhancer_dict = optimizer_dict['enhancer_dict']
                optimizer_dict.pop('enhancer')
                optimizer_dict.pop('enhancer_dict')
            self.optimizer = optimizer(self.net.parameters(), **optimizer_dict)
            self.optim_desc = f'-{type(self.optimizer).__name__}{get_describle(optimizer_dict)}'
            if enhancer and enhancer_dict:
                self.optimizer = enhancer(self.optimizer, **enhancer_dict)
                self.optim_desc = f'{self.optim_desc}-{type(self.optimizer).__name__}{get_describle(enhancer_dict)}'
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters())
            self.optim_desc = f'-{type(self.optimizer).__name__}'

        if lr_scheduler_dict and lr_scheduler_dict.__contains__('lr_scheduler'):
            lr_scheduler = lr_scheduler_dict['lr_scheduler']
            lr_scheduler_dict.pop('lr_scheduler')
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_dict)
            self.lr_scheduler_desc = f'-{type(self.lr_scheduler).__name__}{get_describle(lr_scheduler_dict)}'
        else:
            self.lr_scheduler = None
            self.lr_scheduler_desc = ''
        self.numpy_metric = numpy_metric
        self.monitor = monitor if monitor else 'loss'
        self.monitor_mode = monitor_mode if monitor_mode else 'min'
        self.metrics_dict = metrics_dict if metrics_dict else {}
        self.history = {}

    def fit(self, data, epochs=200, device='cpu', save_dir='./', logger=True, use_compile=False, use_amp=False,
            use_jit=False,
            tb_writer=None, verbose=1):
        time_desc = device.replace(':', '_') + '-' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        trainloader, testloader = data.trainloader, data.testloader
        self.train_desc = f"{data.desc}-{self.desc}{self.loss_desc}{self.optim_desc}{self.lr_scheduler_desc}-epk{epochs}"
        path = f'{save_dir}/{self.train_desc}'
        logger = get_logger(path, f'{time_desc}.log').info if logger else print
        if os.path.exists(path) == False:
            os.makedirs(path)
            logger('Create folder: {}.'.format(path))
        self.net = torch.compile(self.net) if use_compile else self.net
        if use_compile:
            torch.set_float32_matmul_precision('high')
        self.net = torch.jit.script(self.net) if use_jit else self.net
        self.net.to(device)
        if verbose in [1, 2]:
            logger(self.net)
            logger("Start Training ...")
            logger(f'Using metrics: {list(self.metrics_dict.keys())}')
            logger(f'Params: {self.params}\t Trainable Params:{self.params_trainable}')
            print_bar(logger)
        monitor_best = float('inf') if self.monitor_mode == 'min' else -float("inf")
        best_net = copy.deepcopy(self.net)
        best_net_state = copy.deepcopy(self.net.state_dict())
        best_history = {}
        best_preds_trues = {}
        best_epoch = 0
        tqdm_iterator = tqdm(range(1, epochs + 1),
                             desc='[{}-epochs{}]@{}'.format(self.net.desc, epochs, device),
                             leave=True,
                             ncols=250,
                             unit='epochs',
                             disable=(verbose != 1))
        start = time.time()
        for epoch in tqdm_iterator:
            # 1. training loop -------------------------------------------------
            if verbose == 2:
                logger('{}: {}/{} @ {} {}'.format(self.train_desc, epoch, epochs, platform.node(), device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.lr_scheduler:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.history['lr'] = self.history.get('lr', []) + [lr]
            train_metrics, _ = self.trainOneEpoch(
                net=self.net,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                metrics_dict=self.metrics_dict,
                loader=trainloader,
                forward_func=self.forward_func_train_amp if use_amp else self.forward_func_train,
                metric_func=self.metric_func,
                numpy_metric=self.numpy_metric,
                device=device,
                verbose=verbose)
            for name, train_metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [train_metric]
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # 2. validate loop -------------------------------------------------
            val_metrics, pres_trues = self.testOneEpoch(
                net=self.net,
                loss_func=self.loss_func,
                metrics_dict=self.metrics_dict,
                loader=testloader,
                forward_func=self.forward_func_test,
                metric_func=self.metric_func,
                numpy_metric=self.numpy_metric,
                device=device,
                verbose=verbose)
            for name, val_metric in val_metrics.items():
                self.history[name] = self.history.get(name, []) + [val_metric]
            # 3. Record the Best.-------------------------------------------------
            assert self.monitor in self.history.keys()
            compared = self.history[self.monitor][-1] <= monitor_best if self.monitor_mode == 'min' else \
                self.history[self.monitor][-1] >= monitor_best
            if compared:
                if verbose == 2:
                    logger("\n{} update: {} --> {}".format(self.monitor, monitor_best, self.history[self.monitor][-1]))
                best_net = copy.deepcopy(self.net)
                best_net_state = copy.deepcopy(self.net.state_dict())
                best_history = {key: self.history[key][-1] for key in list(self.history.keys())}
                best_epoch = epoch
                best_preds_trues = pres_trues

            monitor_best = min([monitor_best, self.history[self.monitor][-1]]) if self.monitor_mode == 'min' else max(
                [monitor_best, self.history[self.monitor][-1]])
            # 4. print logs -------------------------------------------------
            current_metric = {k: round(self.history[k][-1], 4) for k in self.history.keys()}
            tqdm_iterator.set_postfix(current_metric)
            if verbose == 2:
                infos = {'epoch': f'{epoch}/{epochs}'}
                infos.update(current_metric)
                infos.update({'best_{}'.format(self.monitor): round(monitor_best, 5)})
                infos.update({'best_epoch': best_epoch})

                tb = PrettyTable()
                tb.field_names = infos.keys()
                tb.add_row(infos.values())
                logger(tb)
                print_bar(logger)
            if tb_writer:
                tb_writer.add_scalar(**current_metric, step=epoch)
                tb_writer.add_histogram(self.net, step=epoch)
                tb_writer.flush()
        used_time = time.time() - start
        results_dict = {
            # Save the Bests.
            'desc': self.train_desc,
            'used_epochs': epochs,
            'used_time': used_time,
            'monitor': self.monitor,
            'monitor_mode': self.monitor_mode,
            'best_{}'.format(self.monitor): monitor_best,
            'best_epoch': best_epoch,
            'params': self.params,
            'params_trainable': self.params_trainable,
            'best_history': best_history,
            'val_pred_labels': best_preds_trues['preds'],
            'val_true_labels': best_preds_trues['trues'],
        }
        statedt_dict = {
            'best_net': best_net,
            'best_net_state': best_net_state,
            'final_net_state': copy.deepcopy(self.net.state_dict()),
        }
        history = pandas.DataFrame(self.history)
        history_path = f'{path}/{time_desc}-{self.monitor}_{monitor_best}.his'
        results_path = f'{path}/{time_desc}-{self.monitor}_{monitor_best}.rlt'
        statedt_path = f'{path}/{time_desc}-{self.monitor}_{monitor_best}.sts'
        torch.save(results_dict, results_path)
        torch.save(statedt_dict, statedt_path)
        history.to_pickle(history_path)
        logger("[{}]\tTraining Finished!".format(self.desc))
        logger('[{}]\tBest_{}: {} | {}'.format(self.desc, self.monitor, round(monitor_best, 5), best_history))
        logger('[{}]\tRunning time ({} Epochs): {} Seconds'.format(self.desc, epochs, round(used_time, 5)))
        return results_dict, statedt_dict, history

    def evaluate(self, dataloader, device):
        self.net.to(device)
        return testOneEpoch(
            net=self.net,
            loss_func=self.loss_func,
            metrics_dict=self.metrics_dict,
            loader=dataloader,
            forward_func=self.forward_func_test,
            device=device,
            verbose=2)

    def predict_numpy(self, dataloader, device):
        self.net.to(device)
        self.net.eval()
        true_labels = []
        pred_labels = []
        for features, labels in dataloader:
            features = features.to(device)
            with torch.set_grad_enabled(False):
                out = self.net(features)
                _, out = out.max(1)
                true_labels.extend(list(labels.cpu().numpy()))
                pred_labels.extend(list(out.cpu().numpy()))

        return numpy.array(true_labels), numpy.array(pred_labels)

    def predict_foward(self, dataloader, device):
        self.net.to(device)
        self.net.eval()
        true_labels = []
        preds = []
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                out = self.net(features)
                true_labels.append(labels)
                preds.append(out)
        return torch.cat(preds, dim=0), torch.cat(true_labels, dim=0)

    def predict_prob(self, dataloader, target=0, device='cpu'):
        self.net.to(device)
        self.net.eval()
        true_labels = []
        pred_labels = []
        scores = []
        out_tensors = []
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                out = self.net(features)
                outputs = torch.nn.functional.softmax(out, 1)[:, target].cpu()
                _, preds = torch.max(out, 1)
                true_labels.extend(list(labels.cpu().numpy()))
                pred_labels.extend(list(preds.cpu().numpy()))
                scores.extend(list(outputs.cpu().numpy()))
                out_tensors.append(out)

        return numpy.array(true_labels), numpy.array(pred_labels), numpy.array(scores), torch.cat(out_tensors,
                                                                                                  0).cpu().numpy()


class Trainer(Engine):
    def __init__(self, net, desc=''):
        super().__init__(net, desc)
        self.forward_func_train = forward_train
        self.forward_func_train_amp = functools.partial(forward_train_amp, scaler=torch.amp.GradScaler('cuda'))
        self.forward_func_test = forward_test
        self.metric_func = get_std_loss_and_metrics_results
        self.trainOneEpoch = trainOneEpoch
        self.testOneEpoch = testOneEpoch
