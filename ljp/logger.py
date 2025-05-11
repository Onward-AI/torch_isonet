# coding:utf-8
"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
"""
import os
import logging
from logging import handlers
from torch.utils.tensorboard import SummaryWriter


# 日志根路径
class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            self.writer.add_scalar(head + "/" + k, v, step)

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        self.writer.add_pr_curve(tag, labels, predictions, global_step=global_step,
                                 num_thresholds=num_thresholds, weights=weights, walltime=walltime)

    def add_graph(self, model, inputs):
        self.writer.add_graph(model, (inputs,))

    def add_histogram(self, model, step=None):
        for name, module in model.named_modules():
            for n, param in module.named_parameters():
                self.writer.add_histogram(tag=name + '_' + type(module).__name__ + '_' + n + '_grad', values=param.grad,
                                          global_step=step)
                self.writer.add_histogram(tag=name + '_' + type(module).__name__ + '_' + n + '_data', values=param.data,
                                          global_step=step)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()


def get_logger(log_path, log_filename, level=logging.INFO, when='D', back_count=0):
    """
    :brief  日志记录
    :param log_filename: 日志名称
    :param level: 日志等级
    :param when: 间隔时间:
        S:秒
        M:分
        H:小时
        D:天
        W:每星期（interval==0时代表星期一）
        midnight: 每天凌晨
    :param back_count: 备份文件的个数，若超过该值，就会自动删除
    :return: logger
    """
    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(log_path)
    # log_file_path = os.path.join(log_path, log_filename)
    log_file_path = os.path.join(log_path, log_filename)
    print(log_file_path)
    # log输出格式
    # formatter = logging.Formatter('%(asctime)s | %(pathname)s[line:%(lineno)d] | %(levelname)s: %(message)s')
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    # 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=when,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    # 设置日志输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 添加到logger对象里
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# if __name__ == "__main__":
#     logger = get_logger("my.log")
#     logger.debug("debug test")
#     logger.info("info test")
#     logger.warning("warn test")
#     logger.error("error test")
