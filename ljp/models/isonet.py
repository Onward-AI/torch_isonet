import torch
from ljp.models.isoconv import ISOConv2d
from ljp.activation import AReLU, Smish, TanhExp, Serf

layer_cfgs = {
    '4l004': [4, 'm', 8, 'm', 16, 'm', 32, 'm'],
    '3l008': [8, 'm', 16, 'm', 32, 'm'],
    '4l008': [8, 'm', 16, 'm', 32, 'm', 64, 'm'],
    '5l008': [8, 'm', 16, 'm', 32, 'm', 64, 'm', 128, 'm'],

    '3l016': [16, 'm', 32, 'm', 64, 'm'],
    '4l016': [16, 'm', 32, 'm', 64, 'm', 128, 'm'],
    '5l016': [16, 'm', 32, 'm', 64, 'm', 128, 'm', 256, 'm'],

    '4l020': [20, 'm', 40, 'm', 80, 'm', 160, 'm'],

    '3l032': [32, 'm', 64, 'm', 128, 'm'],
    '4l032': [32, 'm', 64, 'm', 128, 'm', 256, 'm'],
    '5l032': [32, 'm', 64, 'm', 128, 'm', 256, 'm', 256, 'm'],

    '4l040': [40, 'm', 80, 'm', 160, 'm', 320, 'm'],
    '4l060': [60, 'm', 120, 'm', 240, 'm', 480, 'm'],

    '3l064': [64, 'm', 128, 'm', 256, 'm'],
    '4l064': [64, 'm', 128, 'm', 256, 'm', 512, 'm'],
    '5l064': [64, 'm', 128, 'm', 256, 'm', 512, 'm', 1024, 'm'],

    '4l080': [80, 'm', 160, 'm', 320, 'm', 640, 'm'],
    '4l100': [100, 'm', 200, 'm', 400, 'm', 800, 'm'],

    '3l128': [128, 'm', 256, 'm', 512, 'm'],
    '4l128': [128, 'm', 256, 'm', 512, 'm', 1024, 'm'],
    '5l128': [128, 'm', 256, 'm', 512, 'm', 1024, 'm', 2048, 'm'],

    '4l256': [256, 'm', 512, 'm', 1024, 'm', 2048, 'm'],
}
activation_functions = {
    'mish': torch.nn.Mish(),
    'smish': Smish(),
    'relu': torch.nn.ReLU(),
    'arelu': AReLU(),
    'tanhexp': TanhExp(),
    'serf': Serf(),
    'elu': torch.nn.ELU(),
    'celu': torch.nn.CELU(),
    'gelu': torch.nn.GELU(),
    'prelu': torch.nn.PReLU(),
    'selu': torch.nn.SELU(),
    'relu6': torch.nn.ReLU6(),
    'rrelu': torch.nn.RReLU(),
}


class ISONet(torch.nn.Module):
    """
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
    """

    def __init__(self, layer_cfg='4l008', activator='mish', kernel_size=7, inchannels=1, n_output=5):
        super(ISONet, self).__init__()
        self.activator = activator
        self.layer_cfg = layer_cfg
        self.kernel_size = kernel_size
        self.n_output = n_output
        self.inchannels = inchannels
        self.__init()
        self.desc = 'ISONet-{}-{}-ks{}'.format(layer_cfg, activator, kernel_size)

    def __init(self):
        self.Conv2D = ISOConv2d
        cfg_list = layer_cfgs[self.layer_cfg]
        self.conv_padding = 0
        self.ps = 3
        self.p_stride = 3
        self.activation = activation_functions[self.activator]
        self.features = self.__make_layers(cfg_list)
        self.GMP = torch.nn.AdaptiveMaxPool2d((1, 1))

        softmax_classifier = torch.nn.Linear
        num_classifier_in = cfg_list[len(cfg_list) - 2]
        self.classifier = softmax_classifier(num_classifier_in, self.n_output, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = self.GMP(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def __make_layers(self, cfg):
        layers = []
        in_channels = self.inchannels
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'm':
                layers += [torch.nn.MaxPool2d(kernel_size=(1, self.ps), stride=(1, self.p_stride))]
                Mlayers += 1
            else:
                if x == 0:
                    x = 1
                layers += [
                    self.Conv2D(in_channels, x, kernel_size=(1, self.kernel_size), stride=(1, 1), groups=1,
                                padding=(0, self.conv_padding)),
                    torch.nn.BatchNorm2d(x),
                    self.activation]
                in_channels = x
        return torch.nn.Sequential(*layers)
