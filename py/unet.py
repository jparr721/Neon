import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.utils.data


def u_net_layer(input_channels: int, output_channels: int, name: str, kernel_size=2, padding=1, dropout_pct=0.,
                transposed=False, use_batchnorm=True, use_relu=True) -> nn.Sequential:
    """
    U_Network layer with modifications as described in:
    https://arxiv.org/pdf/1810.08217.pdf
    This architecture will also support 3D model discretizations for better approximations of 3D geometric deformation
    :param input_channels: The number of input channels to the convolutional layer
    :param output_channels: The number of output channels to the convolutional layer
    :param name: The name of the layer in the stack of layers
    :param kernel_size: The kernel size in the feature extraction layers
    :param padding: The padding on the feature extractor
    :param dropout_pct: The amount of dropout to use for regularization
    :param transposed: Whether or not we're upsampling (second half of the model)
    :param use_batchnorm: Whether or not to use batchnorm (turns off dropout)
    :param use_relu: Whether or not to use relu activation
    :return: nn.Sequential layer
    """
    block = nn.Sequential()

    if use_relu:
        block.add_module(f"{name}_relu", nn.ReLU(inplace=True))
    else:
        # Leaky relu has a small slope available for negative values instead of slamming them into zeros.
        block.add_module(f"{name}_leaky_relu", nn.LeakyReLU(negative_slope=0.2, inplace=True))

    # We transpose to fix the checkerboard pattern during the upscaling to the output
    if transposed:
        block.add_module(f"{name}_upsampling", nn.Upsample(scale_factor=2, mode="bilinear"))

        # Reduce the kernel size for the upsampling part (decoding back into outputs)
        block.add_module(f"{name}_t_conv",
                         nn.Conv2d(input_channels, output_channels, kernel_size=(kernel_size - 1), stride=2,
                                   padding=padding, bias=True))
    else:
        # Do a regular convolution operation when feature extracting (pre upscaling)
        block.add_module(f"{name}_conv",
                         nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=padding,
                                   bias=True))

    # If selected, we use batch norm instead of dropout due to its regularizing effects. It's useful for smoothing the
    # objective function, leading to better approximations
    if use_batchnorm:
        block.add_module(f"{name}_batchnorm", nn.BatchNorm2d(output_channels))
    else:
        # Otherwise, use good old dropout
        block.add_module(f"{name}_dropout", nn.Dropout2d(dropout_pct, inplace=True))

    return block


def initialize_weights(weights):
    classname = weights.__class__.__name__
    if classname.find('Conv') != -1:
        weights.weight.data.normal_(0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        weights.weight.data.normal_(1.0, 0.02)
        weights.bias.data.fill_(0)


class UNet(nn.Module):
    def __init__(self, dataset_features: int, channel_exponent=6, dropout_pct=0.):
        """
        The UNet is a neural network which performs downsampling and upsampling to produce data of the same shape as
        its inputl
        :param dataset_features: The number of features in the dataset
        :param channel_exponent: The size of the network
        :param dropout_pct: The percentage of dropout to use (between 0 and 1
        """
        super(UNet, self).__init__()

        # Ensure dropout pct is the right value
        assert 0 <= dropout_pct <= 1, "Dropout pct must be between 0 and 1"

        channels = int(2 ** channel_exponent + 0.5)

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("layer_1", nn.Conv2d(
            in_channels=dataset_features,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            padding=1,
            bias=True
        ))

        self.layer_2 = u_net_layer(channels, channels * 2, "encoding_layer_2", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct)
        self.layer_3 = u_net_layer(channels * 2, channels * 2, "encoding_layer_3", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct)
        self.layer_4 = u_net_layer(channels * 2, channels * 4, "encoding_layer_4", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct)
        self.layer_5 = u_net_layer(channels * 4, channels * 8, "encoding_layer_5", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct)
        self.layer_6 = u_net_layer(channels * 8, channels * 8, "encoding_layer_6", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct, kernel_size=2, padding=1)
        self.layer_7 = u_net_layer(channels * 8, channels * 8, "encoding_layer_7", transposed=False, use_batchnorm=True,
                                   use_relu=False, dropout_pct=dropout_pct, kernel_size=2, padding=0)

        self.decoding_layer_7 = u_net_layer(channels * 8, channels * 8, "decoding_layer_7", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct, kernel_size=2, padding=1)
        self.decoding_layer_6 = u_net_layer(channels * 16, channels * 8, "decoding_layer_6", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct, kernel_size=2, padding=1)
        self.decoding_layer_5 = u_net_layer(channels * 16, channels * 4, "decoding_layer_5", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct)
        self.decoding_layer_4 = u_net_layer(channels * 8, channels * 2, "decoding_layer_4", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct)
        self.decoding_layer_3 = u_net_layer(channels * 4, channels * 2, "decoding_layer_3", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct)
        self.decoding_layer_2 = u_net_layer(channels * 4, channels, "decoding_layer_2", transposed=True,
                                            use_batchnorm=True,
                                            use_relu=True, dropout_pct=dropout_pct)

        self.decoding_layer_1 = nn.Sequential()
        self.decoding_layer_1.add_module("decoding_layer_1_relu", nn.ReLU(inplace=True))
        self.decoding_layer_1.add_module("decoding_layer_1_t_conv", nn.ConvTranspose2d(
            in_channels=channels * 2,
            out_channels=dataset_features,
            kernel_size=2,
            stride=2,
            padding=1,
            bias=True
        ))

    def forward(self, x):
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)
        out_6 = self.layer_6(out_5)
        out_7 = self.layer_7(out_6)

        # Run it back
        d_out_6 = self.decoding_layer_7(out_7)
        d_out_6_out_6 = torch.cat([d_out_6, out_6], 1)

        d_out_6 = self.decoding_layer_6(d_out_6_out_6)
        d_out_6_out_5 = torch.cat([d_out_6, out_5], 1)

        d_out_5 = self.decoding_layer_5(d_out_6_out_5)
        d_out_5_out_4 = torch.cat([d_out_5, out_4], 1)

        d_out_4 = self.decoding_layer_4(d_out_5_out_4)
        d_out_4_out_3 = torch.cat([d_out_4, out_3], 1)

        d_out_3 = self.decoding_layer_3(d_out_4_out_3)
        d_out_3_out_2 = torch.cat([d_out_3, out_2], 1)

        d_out_2 = self.decoding_layer_3(d_out_3_out_2)
        d_out_2_out_1 = torch.cat([d_out_2, out_1], 1)

        d_out_1 = self.decoding_layer_1(d_out_2_out_1)
        return d_out_1

    @property
    def n_parameters(self):
        nn_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in nn_parameters])
