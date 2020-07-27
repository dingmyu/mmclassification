from .alexnet import AlexNet
from .lenet import LeNet5
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetv3
from .regnet import RegNet
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .hrnet import HighResolutionNet
from .hrnet_origin import HighResolutionNetOri
from .hrnet_simple import HighResolutionNetSim
from .hrnet_dilated import HighResolutionNetDia

__all__ = [
    'LeNet5', 'AlexNet', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNetV1d', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetv3',
    'HighResolutionNet', 'HighResolutionNetOri', 'HighResolutionNetSim', 'HighResolutionNetDia'
]
