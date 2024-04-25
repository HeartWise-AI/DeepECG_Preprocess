import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_backbone_config(variant):
    configs = {

        'b0_v1': {'width_coefficient': 1.0, 'depth_coefficient': 1.0},
        'b1_v1': {'width_coefficient': 1.0, 'depth_coefficient': 1.1},
        'b2_v1': {'width_coefficient': 1.1, 'depth_coefficient': 1.2},
        'b3_v1': {'width_coefficient': 1.2, 'depth_coefficient': 1.4},
        'b4_v1': {'width_coefficient': 1.4, 'depth_coefficient': 1.8},
        'b5_v1': {'width_coefficient': 1.6, 'depth_coefficient': 2.2},
        'b6_v1': {'width_coefficient': 1.8, 'depth_coefficient': 2.6},
        'b7_v1': {'width_coefficient': 2.0, 'depth_coefficient': 3.1},

        'b0_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.0},
        'b1_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.1},
        'b2_v2': {'width_coefficient': 1.1, 'depth_coefficient': 1.2},
        'b3_v2': {'width_coefficient': 1.2, 'depth_coefficient': 1.4},
        'b4_v2': {'width_coefficient': 1.4, 'depth_coefficient': 1.8},
        'b5_v2': {'width_coefficient': 1.6, 'depth_coefficient': 2.2},
        'b6_v2': {'width_coefficient': 1.8, 'depth_coefficient': 2.6},
        'b7_v2': {'width_coefficient': 2.0, 'depth_coefficient': 3.1},
        's_v2':  {'width_coefficient': 1.0, 'depth_coefficient': 2.0},
        'm_v2':  {'width_coefficient': 1.1, 'depth_coefficient': 2.1},
        'l_v2':  {'width_coefficient': 1.2, 'depth_coefficient': 2.2},
    }
    return configs[variant]

def get_activation(name='relu'):
    activations = {
        'relu': nn.ReLU,
        'swish': nn.SiLU,
        'mish': nn.Mish,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'leaky_relu': nn.LeakyReLU,
    }
    return activations[name](inplace=True) if name != 'gelu' else activations[name]()


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim, activation_func=nn.ReLU(inplace=True)):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            activation_func,
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
        return x

class FusedMBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0):
        super(FusedMBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)

        self.fused_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            activation_func,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()
        self.se = SEBlock(out_channels, out_channels // se_ratio, activation_func) if use_se else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.fused_conv(x)
        x = self.se(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

class MBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=1, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0):
        super(MBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)
        mid_channels = int(in_channels * expansion)

        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            activation_func,
        ) if expansion > 1 else nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size, stride, kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            activation_func,
        )


        self.se = SEBlock(mid_channels, max(1, int(mid_channels // se_ratio)), activation_func) if use_se else nn.Identity()
        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

class EfficientNet1DV2(nn.Module):
    def __init__(self, variant='s', input_channels=12, num_classes=77, activation='swish', se_ratio=4, base_depths=None, base_channels=None, expansion_factors=None, stochastic_depth_prob=0.0, dropout_rate=0.0, use_se=True, kernel_sizes=None, strides=None):
        if base_depths is None:
            base_depths = [1, 2, 2, 3, 3, 4, 1]
        if base_channels is None:
            base_channels = [32, 16, 24, 40, 80, 112, 192]
        if expansion_factors is None:
            expansion_factors = [1, 6, 6, 6, 6, 6, 6]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3]
        if strides is None:
            strides = [1, 2, 2, 2, 1, 2, 2, 1]
        super(EfficientNet1DV2, self).__init__()
        config = get_backbone_config(variant)
        width_coefficient, depth_coefficient = config['width_coefficient'], config['depth_coefficient']
        print(activation)
        # Define default parameters for different variants
        #kernel_sizes, strides, se_ratio, base_depths, base_channels, expansion_factors = self._get_default_parameters(variant)

        # Apply coefficients to channels and depths
        channels = [max(1, int(c * width_coefficient)) for c in base_channels]
        depths = [max(1, math.ceil(d * depth_coefficient)) for d in base_depths]

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm1d(channels[0]),
            get_activation(activation),
        )

        # Constructing the blocks
        self.features = self._make_layers(variant, channels, depths, kernel_sizes, strides, expansion_factors, se_ratio, activation, stochastic_depth_prob, dropout_rate, use_se)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], num_classes),
        )

    def _get_default_parameters(self, variant):
        if variant in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
            kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3]
            strides = [1, 2, 2, 2, 1, 2, 2, 1]
            se_ratio = 4
            base_depths = [1, 2, 2, 3, 3, 4, 1]
            base_channels = [32, 16, 24, 40, 80, 112, 192]
            expansion_factors = [1, 6, 6, 6, 6, 6, 6]
        else:  # Assuming 's', 'm', 'l' follow a different pattern
            kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3]
            strides = [2, 1, 2, 2, 2, 1, 2]
            se_ratio = 4
            base_depths = [1, 2, 4, 4, 6, 9, 15]
            base_channels = [24, 24, 48, 64, 128, 160, 256]
            expansion_factors = [1, 1, 4, 4, 4, 6, 6]
        return kernel_sizes, strides, se_ratio, base_depths, base_channels, expansion_factors

    def _make_layers(self, variant, channels, depths, kernel_sizes, strides, expansion_factors, se_ratio, activation, stochastic_depth_prob, dropout_rate, use_se):
        layers = []
        in_channels = channels[0]
        for i, (out_channels, num_blocks) in enumerate(zip(channels[1:], depths)):
            if 'v2' in variant:
                for j in range(num_blocks):
                    stride = strides[i] if j == 0 else 1
                    if i < 3:  # Use FusedMBConv1d for the first 3 stages
                        layers.append(FusedMBConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i], stride=strides[i], activation=activation, use_se=use_se, se_ratio=se_ratio[i], dropout_rate=dropout_rate, stochastic_depth_prob=stochastic_depth_prob))
                    else:  # Use MBConv1d for later stages
                        layers.append(MBConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i], stride=strides[i], expansion=expansion_factors[i], activation=activation, use_se=use_se, se_ratio=se_ratio[i], dropout_rate=dropout_rate, stochastic_depth_prob=stochastic_depth_prob))
                    in_channels = out_channels

            else:
                for j in range(num_blocks):
                    stride = strides[i] if j == 0 else 1
                    layers.append(MBConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i], stride=strides[i], expansion=expansion_factors[i], activation=activation, use_se=use_se, se_ratio=se_ratio[i], dropout_rate=dropout_rate, stochastic_depth_prob=stochastic_depth_prob))
                    in_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
