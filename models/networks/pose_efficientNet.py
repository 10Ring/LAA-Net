#-*- coding: utf-8 -*-
import math
import sys
import os
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

from ..builder import MODELS, build_model
from .efficientNet import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size,
    url_map_advprop,
    url_map
)
from .common import (
    InceptionBlock,
    conv_block,
    BN_MOMENTUM,
    SELayer
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


@MODELS.register_module()
class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # Heatmap Decoder Construction
        if self._global_params.include_hm_decoder:
            print("Constructing the heatmap Decoder!")
            self.efpn = self._global_params.efpn
            self.tfpn = self._global_params.tfpn

            assert not (self.efpn and self.tfpn), "Only one of E-FPN or FPN is intergrated!"

            self.se_layer = self._global_params.se_layer
            # self.hm_decoder_filters = [1792, 448, 160, 56] if self.fpn else [1792, 256, 256, 128]
            self.hm_decoder_filters = [1792, 448, 160, 56]
            num_kernels = [4, 4, 4, 4] if (self.efpn or self.tfpn) else [4, 4, 4]
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._sigmoid = nn.Sigmoid()
            self._relu = nn.ReLU(inplace=True)
            self._relu1 = nn.ReLU(inplace=False)
            self.deconv_with_bias = False
            if self._global_params.use_c3:
                self.inception_block = InceptionBlock(112, 112, stride=1, pool_size=3)
            else:
                self.inception_block = InceptionBlock(56, 56, stride=1, pool_size=3)
            self.heads = self._global_params.heads
            n_deconv = len(self.hm_decoder_filters)
            self.fpn_layers = [self._global_params.use_c51, self._global_params.use_c4, self._global_params.use_c3]
            
            if self.efpn or self.tfpn:
                for idx in range(n_deconv):
                    in_decod_filters = self.hm_decoder_filters[idx]
                    
                    if idx == 0:
                        out_decod_filters = self.hm_decoder_filters[idx+1]
                        deconv = nn.Sequential(
                            conv_block(in_decod_filters, out_decod_filters, (3,3), stride=1, padding=1),
                        )
                    else:
                        in_decod_filters = in_decod_filters*2 if self.fpn_layers[idx-1] else in_decod_filters
                        kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[idx])

                        if idx+1 < n_deconv:
                            out_decod_filters = self.hm_decoder_filters[idx+1]
                            deconv = nn.Sequential(
                                conv_block(in_decod_filters, out_decod_filters, (3,3), stride=1, padding=1),
                                nn.ConvTranspose2d(
                                    in_channels=out_decod_filters,
                                    out_channels=out_decod_filters,
                                    kernel_size=kernel,
                                    stride=2,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=self.deconv_with_bias),
                                nn.BatchNorm2d(out_decod_filters, momentum=BN_MOMENTUM),
                            )
                        else:
                            out_decod_filters = in_decod_filters
                            deconv = nn.Sequential(
                                self.inception_block,
                                nn.ConvTranspose2d(
                                    in_channels=out_decod_filters,
                                    out_channels=out_decod_filters,
                                    kernel_size=kernel,
                                    stride=2,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=self.deconv_with_bias),
                                nn.BatchNorm2d(out_decod_filters, momentum=BN_MOMENTUM),
                            )
                            
                            # In case of using C2, this conv to apply to C2 features to get the same filters of the last deconv
                            if self._global_params.use_c2:
                                self.conv_c2 = conv_block(32, out_decod_filters, (3,3), stride=1, padding=1)
                    if self.se_layer:
                        se = SELayer(channel=out_decod_filters*2)
                        self.__setattr__(f'se_layer_{idx+1}', se)
                        
                    self.__setattr__(f'deconv_{idx+1}', deconv)
            else:
                self.deconv_layers = self._make_deconv_layer(
                    len(num_kernels),
                    self.hm_decoder_filters,
                    num_kernels,
                )

            for head, num_output in self.heads.items():
                head_conv = int(self._global_params.head_conv)
                num_output = int(num_output)
                if self._global_params.use_c2:
                    assert self._global_params.efpn or self._global_params.tfpn, "FPN Design must be set active!"
                    assert self._global_params.use_c3, "C3 must be utilized for FPN intergration of C2"
                    in_head_filters = self.hm_decoder_filters[-1]*4
                elif self._global_params.use_c3:
                    in_head_filters = self.hm_decoder_filters[-1]*2
                else:
                    in_head_filters = self.hm_decoder_filters[-1]

                if head_conv > 0:
                    if head != 'cls':
                        fc = nn.Sequential(
                            nn.Conv2d(in_head_filters, head_conv,
                            kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(head_conv),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(head_conv, num_output, 
                            kernel_size=1, stride=1, padding=0)
                        )
                    else:
                        fc = nn.Sequential(
                            nn.Conv2d(in_head_filters, head_conv, kernel_size=3,
                                        padding=1, bias=True),
                            nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                            # nn.Conv2d(head_conv, num_output, kernel_size=1,
                            #             stride=1, padding=0, bias=True),
                            # nn.BatchNorm2d(num_output),
                            # nn.ReLU(inplace=True),
                            # nn.AdaptiveMaxPool2d(head_conv//4),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            # nn.Linear((head_conv//4)**2, head_conv, bias=True),
                            # nn.BatchNorm1d(head_conv, momentum=BN_MOMENTUM),
                            # nn.ReLU(inplace=True),
                            nn.Linear(head_conv, num_output, bias=True),
                            # nn.Sigmoid(),
                            # nn.Softmax(dim=-1)
                        )
                else:
                    fc = nn.Conv2d(
                        in_channels=in_head_filters,
                        out_channels=num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                self.__setattr__(head, fc)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == (len(num_filters) - 1), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            in_planes = num_filters[i]
            out_planes = num_filters[i+1]

            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias),
                nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
            )

        return nn.Sequential(*layers)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print('Prev', prev_x.size())
            # print('X', x.size())
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        # x = self.extract_features(inputs)
        endpoints = self.extract_endpoints(inputs)
        x1 = endpoints['reduction_6']
        x2 = endpoints['reduction_5']
        x3 = endpoints['reduction_4']
        x4 = endpoints['reduction_3']
        x5 = endpoints['reduction_2']
        x = x1
        
        if self._global_params.include_top:
            # Pooling and final linear layer
            x = self._avg_pooling(x)

            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
            return x

        if self._global_params.include_hm_decoder:
            x1 = self._dropout(x1)
            x2 = self._dropout(x2)
            x3 = self._dropout(x3)
            x4 = self._dropout(x4)

            if self.efpn:
                assert self._global_params.use_c51, "C51 must be utilized for FPN intergration"
                
                x = self.__getattr__('deconv_1')(x1)
                
                if self._global_params.use_c51:
                    x_weighted = self._sigmoid(x)
                    x_inv = torch.sub(1, x_weighted, alpha=1)
                    x2_ = torch.multiply(x_inv, x2)
                    x = torch.cat([x, x2_], dim=1)
                    
                    if self.se_layer:
                        x = self.__getattr__('se_layer_1')(x)
                else:
                    x = self._relu(x)
                
                x = self.__getattr__('deconv_2')(x)
                
                if self._global_params.use_c4:
                    x_weighted = self._sigmoid(x)
                    x_inv = torch.sub(1, x_weighted, alpha=1)
                    x3_ = torch.multiply(x_inv, x3)
                    x = torch.cat([x, x3_], dim=1)
                    
                    if self.se_layer:
                        x = self.__getattr__('se_layer_2')(x)
                else:
                    x = self._relu(x)
                
                x = self.__getattr__('deconv_3')(x)
                
                if self._global_params.use_c3:
                    assert self._global_params.use_c4, "C4 must be utilized for FPN intergration of C3"
                    
                    x_weighted = self._sigmoid(x)
                    x_inv = torch.sub(1, x_weighted, alpha=1)
                    x4_ = torch.multiply(x_inv, x4)
                    x = torch.cat([x, x4_], dim=1)
                    
                    if self.se_layer:
                        x = self.__getattr__('se_layer_3')(x)
                else:
                    x = self._relu(x)
                
                x = self.__getattr__('deconv_4')(x)
                
                if not self._global_params.use_c2:
                    x = self._relu(x)
                else:
                    assert self._global_params.use_c3, "C3 must be utilized for FPN intergration of C2"
                    
                    x5 = self._dropout(x5)
                    x5_ = self.conv_c2(x5)
                    x_weighted = self._sigmoid(x)
                    x_inv = torch.sub(1, x_weighted, alpha=1)
                    x5_ = torch.multiply(x_inv, x5_)
                    x = torch.cat([x, x5_], dim=1)
                    
                    if self.se_layer:
                        x = self.__getattr__('se_layer_4')(x)
            elif self.tfpn:
                assert self._global_params.use_c51, "C51 must be utilized for FPN intergration"
                x = self.__getattr__('deconv_1')(x1)
                x = self._relu1(x)
                x = torch.cat([x, x2], dim=1)
                
                x = self.__getattr__('deconv_2')(x)
                if not self._global_params.use_c4:
                    x = self._relu1(x)
                else:
                    x = torch.cat([x, x3], dim=1)
                
                x = self.__getattr__('deconv_3')(x)
                if not self._global_params.use_c3:
                    x = self._relu1(x)
                else:
                    assert self._global_params.use_c4, "C4 must be utilized for FPN intergration of C3"
                    x = torch.cat([x, x4], dim=1)
                
                x = self.__getattr__('deconv_4')(x)
                if not self._global_params.use_c2:
                    x = self._relu(x)
                else:
                    assert self._global_params.use_c3, "C3 must be utilized for FPN intergration of C2"
                    x5 = self._dropout(x5)
                    x5 = self.conv_c2(x5)
                    x = self._relu1(x)
                    x = torch.cat([x, x5], dim=1)
            else:
                x = self.deconv_layers(x1)
                
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(x)
            
            return [ret]

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.
        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.
        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=((num_classes == 1000) and (model._global_params.include_top)), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    

@MODELS.register_module()
class PoseEfficientNet(EfficientNet):
    def __init__(self, model_name, in_channels=3, **override_params):
        self.model_name = model_name
        self.in_channels = in_channels

        # Initialize Parent Class
        super()._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        super().__init__(blocks_args, global_params)
    
    @classmethod
    def from_name(cls, model_name, in_channels, **override_params):
        return NotImplemented
    
    @classmethod
    def from_pretrained(cls, model_name, weights_path, advprop, in_channels, num_classes, **override_params):
        return NotImplemented

    def _change_in_channels(self, in_channels):
        return NotImplemented
    
    def init_weights(self, pretrained=False, advprop=False, verbose=True):
        if pretrained:
            url_map_ = url_map_advprop if advprop else url_map
            state_dict = model_zoo.load_url(url_map_[self.model_name])
            self.load_state_dict(state_dict, strict=False)

        # Initialize weights for Deconvolution Layer
        if self._global_params.include_hm_decoder:
            if self.efpn or self.tfpn:
                deconv_layers = [self.deconv_1, self.deconv_2, self.deconv_3, self.deconv_4]
            else:
                deconv_layers = self.deconv_layers
                
            for layer in deconv_layers:
                for _, m in layer.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            # Init head parameters
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                # nn.init.normal_(m.weight, std=0.001)
                                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                                m.weight.data.normal_(0, math.sqrt(2. / n))
                                nn.init.constant_(m.bias, 0)
        
        self._change_in_channels(in_channels=self.in_channels)
        if verbose:
            print('Loaded pretrained weights for {}'.format(self.model_name))


if __name__ == '__main__':
    cfg = dict(type='PoseEfficientNet',
               model_name='efficientnet-b4', 
               include_top=False, 
               include_hm_decoder=True, 
               head_conv=64,
               heads={'hm':1, 'cls':1, 'cstency':256},
               use_c2=True)
    model = build_model(cfg, MODELS)
    model.init_weights(pretrained=True)
    model.eval()
    inputs = torch.rand((1, 3, 384, 384))
    
    for i, (n, p) in enumerate(model.named_parameters()):
        print(i, n)
    
    # To show the whole pose EFN model outputs shape
    x = model(inputs)[0]
    for head in x.keys():
        print(f'{head} shape is --- {x[head].shape}')
    
    # To show the endpoints features shape 
    # endpoints = model.extract_endpoints(inputs)
    # for k in endpoints.keys():
    #     print(endpoints[k].shape)
