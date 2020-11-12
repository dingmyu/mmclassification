import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument('--net_params', type=str, default='')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)

    if args.net_params:
        tag, input_channels, block1, block2, block3, block4, last_channel = args.net_params.split('-')
        input_channels = [int(item) for item in input_channels.split('_')]
        block1 = [int(item) for item in block1.split('_')]
        block2 = [int(item) for item in block2.split('_')]
        block3 = [int(item) for item in block3.split('_')]
        block4 = [int(item) for item in block4.split('_')]
        last_channel = int(last_channel)

        inverted_residual_setting = []
        for item in [block1, block2, block3, block4]:
            for _ in range(item[0]):
                inverted_residual_setting.append([item[1], item[2:-int(len(item)/2-1)], item[-int(len(item)/2-1):]])

        cfg.model.backbone.input_channel = input_channels
        cfg.model.backbone.inverted_residual_setting = inverted_residual_setting
        cfg.model.backbone.last_channel = last_channel
        cfg.model.head.in_channels = last_channel

    model = build_classifier(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
