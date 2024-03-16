
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Configuration for different ResNet architectures
resnet_configs = {
    'resnet18': {'block_channels': [64, 64, 128, 256, 512], 'num_blocks': [2, 2, 2, 2]},
    'resnet18_ECG': { 'block_channels': [64, 1024, 256, 64, 16], 'num_blocks': [2, 2, 2, 2]},
    'resnet34': {'block_channels': [64, 64, 128, 256, 512], 'num_blocks': [3, 4, 6, 3]},
    'resnet50': {'block_channels': [64, 256, 512, 1024, 2048], 'num_blocks': [3, 4, 6, 3]},
    'resnet101': {'block_channels': [64, 256, 512, 1024, 2048], 'num_blocks': [3, 4, 23, 3]},
    'resnet152': {'block_channels': [64, 256, 512, 1024, 2048], 'num_blocks': [3, 8, 36, 3]},
    'resnet200': {'block_channels': [64, 256, 512, 1024, 2048], 'num_blocks': [3, 24, 48, 3]},

}


class EMA:
    """
    Exponential Moving Average (EMA) class for model parameters.

    Args:
        model (nn.Module): The model whose parameters will be averaged.
        decay (float, optional): The decay rate for the moving average. Defaults to 0.999.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        """
        Register the model's parameters for EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update the EMA shadow parameters based on the current model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data


class Mish(nn.Module):
    """
    The Mish activation function.
    
    Mish is a non-linear activation function that is known to perform well in deep learning models.
    It is defined as the element-wise multiplication of the input with the hyperbolic tangent of the softplus function.
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class ResNetBlock(nn.Module):
    """
    Residual Network (ResNet) block implementation.

    Args:
        model (str): The ResNet model type.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        stride (int): Stride value for the convolutional layers. Default is 1.
        activation (str): Activation function to use. Default is 'relu'.
        dropout (float): Dropout probability. Default is 0.0.
        stochastic_depth (float): Probability of skipping the block during training. Default is 0.0.
        use_bias (bool): Whether to use bias in the convolutional layers. Default is True.
    """

    def __init__(self, model, in_channels, out_channels, kernel_size=3, stride=1, activation='relu',
                 dropout=0.0, stochastic_depth=0.0, use_bias=True):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=use_bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = self.get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.stochastic_depth = stochastic_depth

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=use_bias)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            padding = kernel_size // 2 - 1  # Adjust the padding to match the spatial dimensions
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=use_bias),
                nn.BatchNorm1d(out_channels),
                nn.AvgPool1d(kernel_size=stride)
            )
        else:
            self.shortcut = nn.Identity()
        # Initialize convolutions with He uniform
        self.apply(self.init_weights)

        self.model = model

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=0, std=0.01)
            init.constant_(m.bias, 0)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'mish':
            return Mish()
        elif activation == 'swish':
            return Swish()
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def forward(self, x):
        """
        Forward pass of the ResNetBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = self.shortcut(x)

        if self.training and self.stochastic_depth > 0.0:
            if torch.rand(1).item() < self.stochastic_depth:
                return identity  # Skip the block

        residual = self.shortcut(x)
        #if self.shortcut == nn.Identity():
        #residual = self.shortcut[0](residual)
        if self.model in ['resnet18','resnet34']:
            residual = F.avg_pool1d(residual, kernel_size=residual.size(-1))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual

        x = self.activation(x)
        x = self.dropout(x)

        return x


class ResNet1D(nn.Module):
    """
    ResNet1D is a 1D convolutional neural network model based on the ResNet architecture.

    Args:
        input_channels (int): Number of input channels.
        architecture (str): ResNet architecture type (default: 'resnet18').
        activation (str): Activation function to use (default: 'relu').
        dropout (float): Dropout probability (default: 0.0).
        use_batchnorm_padding_before_conv1 (bool): Whether to use batch normalization and padding before the first convolutional layer (default: False).
        use_padding_pooling_after_conv1 (bool): Whether to use padding and pooling after the first convolutional layer (default: False).
        stochastic_depth (float): Stochastic depth probability (default: 0.0).
        kernel_sizes (list): List of kernel sizes for each convolutional layer (default: None).
        strides (list): List of stride values for each convolutional layer (default: None).
        use_bias (bool): Whether to use bias in convolutional layers (default: True).
        out_neurons (int): Number of output neurons (default: 92).
        out_activation (str): Activation function for the final layer (default: 'sigmoid').
        model_width (int): Width multiplier for the model (default: 1).
        use_ema (bool): Whether to use exponential moving average (EMA) for model parameters (default: True).
        **kwargs: Additional keyword arguments to configure the ResNet architecture.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer after the first convolutional layer.
        activation (nn.Module): Activation function.
        dropout (nn.Dropout): Dropout layer.
        blocks (nn.Sequential): Sequence of ResNet blocks.
        final_layer (nn.Linear): Final fully connected layer.
        final_activation (nn.Module): Activation function for the final layer.
        ema (EMA): Exponential moving average for model parameters.

    Methods:
        forward(x): Forward pass of the model.
        apply_ema(): Update EMA parameters after each training step.
        set_to_eval_mode(): Set the model to evaluation mode using EMA parameters.
        set_to_train_mode(): Set the model back to training mode.
    """
    # Rest of the code...
class ResNet1D(nn.Module):


    def __init__(self, input_channels, architecture='resnet18', activation='relu', dropout=0.0,
                 use_batchnorm_padding_before_conv1=False, use_padding_pooling_after_conv1=False,
                 stochastic_depth=0.0, kernel_sizes=None, strides=None, use_bias=True, out_neurons=92,out_activation='sigmoid',model_width=1, use_ema=True, **kwargs):
        super(ResNet1D, self).__init__()
        
        config = resnet_configs[architecture]
        config.update(kwargs)

        if use_batchnorm_padding_before_conv1:
            self.bn_padding = nn.BatchNorm1d(input_channels)
            self.zero_padding_before_conv1 = nn.ConstantPad1d((0, 1), 0)

        self.stochastic_depth = stochastic_depth
        print(config['block_channels'])

        if model_width != 1:
            config['block_channels'] =  [int(ch * model_width) for ch in config['block_channels']]
            #config['block_channels'] =  [max(1, ch) for ch in config['block_channels']]

        print(config['block_channels'])
        self.conv1 = nn.Conv1d(input_channels, config['block_channels'][0],
                               kernel_size=kernel_sizes[0], stride=strides[0],
                               padding=kernel_sizes[0] // 2, bias=use_bias)
        self.bn1 = nn.BatchNorm1d(config['block_channels'][0])

        self.activation = self.get_activation(activation)

        self.dropout = nn.Dropout(p=dropout)

        if use_padding_pooling_after_conv1:
            self.zero_padding_after_conv1 = nn.ConstantPad1d((0, 1), 0)
            self.pooling_after_conv1 = nn.MaxPool1d(strides[1] + 1, stride=strides[1], padding=1)

        self.blocks = nn.Sequential(
            *[ResNetBlock(architecture,config['block_channels'][i], config['block_channels'][i + 1],
                          kernel_size=kernel_sizes[i + 1], stride=strides[i + 1],
                          activation=activation, dropout=dropout, stochastic_depth=stochastic_depth,
                          use_bias=use_bias)
              for i in range(len(config['block_channels']) - 1)]
        )

        # Initialize convolutions with He uniform
        self.apply(self.init_weights)

        if out_neurons is not None:
            self.final_layer = nn.Linear(config['block_channels'][-1], out_neurons)
            if out_activation is not None:
                self.final_activation = self.get_activation(out_activation)

        # Initialize EMA
        if use_ema:
            self.ema = EMA(self)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=0, std=0.01)
            init.constant_(m.bias, 0)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'mish':
            return Mish()
        elif activation == 'swish':
            return Swish()
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def forward(self, x):
            """
            Forward pass of the ResNet model.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor.
            """
            if hasattr(self, 'bn_padding') and hasattr(self, 'zero_padding_before_conv1'):
                x = self.bn_padding(x)
                x = self.zero_padding_before_conv1(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.dropout(x)

            if hasattr(self, 'zero_padding_after_conv1') and hasattr(self, 'pooling_after_conv1'):
                x = self.zero_padding_after_conv1(x)
                x = self.pooling_after_conv1(x)

            x = self.blocks(x)

            if hasattr(self, 'final_layer'):
                x = x.mean(dim=-1)  # Global average pooling
                x = self.final_layer(x)
                #if hasattr(self, 'final_activation'):
                    #x = self.final_activation(x)
            return x

    def apply_ema(self):
        """
        Applies Exponential Moving Average (EMA) to update the model parameters after each training step.
        """
        # Update EMA parameters after each training step
        self.ema.update()

    def set_to_eval_mode(self):
            """
            Sets the model to evaluation mode using EMA parameters.
            
            This method sets the model to evaluation mode by updating the model's parameters
            with the exponential moving average (EMA) parameters.
            """
            
            # Set the model to evaluation mode using EMA parameters
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data = self.ema.shadow[name]

    def set_to_train_mode(self):
        # Set the model back to training mode
        self.train()

# TODO
# allow 2d and max vs avg pool



if __name__ == "__main__":
    # model = ResNet1D(
    #     input_channels=12,
    #     architecture='resnet18',
    #     activation='relu',
    #     dropout=0.8,
    #     use_batchnorm_padding_before_conv1=True,
    #     use_padding_pooling_after_conv1=True,
    #     stochastic_depth=0.0,
    #     kernel_sizes=[49, 7, 7, 5, 3],  # Custom kernel sizes for each stage
    #     strides=[2, 2, 2, 2, 2],  # Custom strides for each stage
    #     use_bias=True,
    #     use_2d = False,
    #     out_neurons=1,
    # )
    
    
    model = ResNet1D(
        input_channels=12,
        architecture='resnet18_ECG',
        activation='relu',
        dropout=0.8,
        use_batchnorm_padding_before_conv1=True,
        use_padding_pooling_after_conv1=True,
        stochastic_depth=0.0,
        kernel_sizes=[16, 16, 16, 16, 16],  # Custom kernel sizes for each stage
        strides=[1, 1, 1, 1, 1],  # Custom strides for each stage
        use_bias=False,
        use_2d = False,
        out_neurons=1,)
    
    print(model)
    
    
