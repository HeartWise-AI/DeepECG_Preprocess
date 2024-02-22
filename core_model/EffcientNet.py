import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print('printlayer')
        print(x.shape)
        return x

class AdaptiveLinearModel(nn.Module):
    def __init__(self, output_features):
        super(AdaptiveLinearModel, self).__init__()
        self.output_features = output_features
        self.linear = None  # Initialize without a defined linear layer

    def forward(self, x):
        if self.linear is None:
            # Determine the number of input features from the input tensor
            input_features = x.size(-1)
            # Dynamically create the linear layer based on the input size
            self.linear = nn.Linear(input_features, self.output_features)
            # If using GPU, ensure the newly created layer is moved to the same device as the input
            self.linear = self.linear.to(x.device)
        
        return self.linear(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation(name='relu'):
    """Returns the activation function by name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'swish': nn.SiLU(inplace=True),  # Swish is also known as SiLU
        'mish': nn.Mish(inplace=True),
        'selu': nn.SELU(inplace=True),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True)}

    return activations[name]

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for 1D inputs with flexible reduction ratio and activation function."""
    def __init__(self, in_channels, reduction_ratio, activation_func):
        super(SEBlock, self).__init__()
        reduced_dim = max(1, in_channels // reduction_ratio)  # Calculate reduced dimension based on the provided ratio
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
    def forward(self, x):
        if self.training and self.drop_prob > 0:
            binary_tensor = torch.rand([x.size(0), 1, 1], dtype=x.dtype, device=x.device) < (1 - self.drop_prob)
            return x * binary_tensor
        return x


class MBConv1d(nn.Module):
    """1D Mobile Inverted Residual Bottleneck Block with SE and Stochastic Depth."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, se_ratio, dropout_rate, stochastic_depth_prob, activation_func, expansion=1, use_se=True):
        super(MBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        mid_channels = in_channels * expansion
        self.stochastic_depth_prob = stochastic_depth_prob

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

        self.se = SEBlock(mid_channels, se_ratio, activation_func) if use_se else nn.Identity()

        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        #print(self.expansion)
        #print(self.in_channels)
        #print(self.mid_channels)

        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.dropout(x)

        if self.use_residual:
            if self.training and torch.rand(1).item() < (1 - self.stochastic_depth_prob):
                return identity
            else:
                return x + identity
        return x
        
class EfficientNet1D(nn.Module):
    """EfficientNet 1D model with dynamic depth, width scaling, and customizable kernel sizes, strides, and input channels."""
    def __init__(self, variant='b0', input_channels=12, stochastic_depth_prob=0, dropout_rate=0.2, activation='swish', width_multiplier=1.0, kernel_sizes=None, strides=None, se_ratio=None, expansion_factors=None, base_depths=None, base_channels=None):
        super(EfficientNet1D, self).__init__()
        # Default kernel sizes and strides for EfficientNet-B0 if not specified
        default_kernel_sizes = [3, 3, 3, 5, 3, 5, 5, 3]
        default_strides = [1, 1, 2, 2, 2, 1, 2, 1]
        default_se_ratio = [4, 4, 4, 4, 4, 4, 4]
        default_base_depths = [1, 2, 2, 3, 3, 4, 1]
        default_base_channels = [32, 16, 24, 40, 80, 112, 192] 
        default_expansion_factors = [1, 6, 6, 6, 6, 6, 6]

        self.activation_func = get_activation(activation)

        #kernel_sizes = default_kernel_sizes
        #strides = default_strides
        #se_ratio = default_se_ratio
        #base_depths = default_base_depths
        #base_channels = default_base_channels
        #expansion_factors = default_expansion_factors

        kernel_sizes = kernel_sizes if kernel_sizes is not None else default_kernel_sizes
        strides = strides if strides is not None else default_strides
        se_ratio =  se_ratio if se_ratio is not None else default_se_ratio
        base_depths = base_depths if base_depths is not None else default_base_depths
        base_channels = base_channels if base_channels is not None else default_base_channels
        expansion_factors = expansion_factors if expansion_factors is not None else default_expansion_factors

        # Base settings for EfficientNet-B0, adjustable via variant and width_multiplier
    
        
        width_coefficient, depth_coefficient = self._get_coefficients(variant)
        # Adjust base_channels[0] to match input_channels for the first layer
        channels = [max(1, int(c * width_coefficient * width_multiplier)) for c in base_channels]
        depths = [max(1, math.ceil(d * depth_coefficient)) for d in base_depths]

        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=1),
            nn.BatchNorm1d(channels[0]),
            self.activation_func,
        )

        # Build the model
        layers = []
        in_channels = channels[0]
        for i, (out_channels, depth, ks, s, exp, se_r) in enumerate(zip(channels, depths, kernel_sizes[1::], strides[1::], expansion_factors, se_ratio)):

            for j in range(depth):
                layers.append(MBConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=exp,
                    kernel_size=ks,
                    use_se=True,
                    stride=s,
                    se_ratio=se_r,
                    activation_func=self.activation_func,
                    dropout_rate=dropout_rate,
                    stochastic_depth_prob=stochastic_depth_prob,  # Placeholder for stochastic depth calculation
                ))
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_channels,77),  # Example: Adjust this as needed for your output classes
        )

    def _get_coefficients(self, variant):
        variants = {
            'b0': (1.0, 1.0),
            'b1': (1.0, 1.1),
            'b2': (1.1, 1.2),
            'b3': (1.2, 1.4),
            'b4': (1.4, 1.8),
            'b5': (1.6, 2.2),
            'b6': (1.8, 2.6),
            'b7': (2.0, 3.1),
        }
        return variants.get(variant, (1.0, 1.0))
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

