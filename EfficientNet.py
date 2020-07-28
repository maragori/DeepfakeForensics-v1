###############################################################################################
#       Code written by Luke Melas https://github.com/lukemelas/EfficientNet-PyTorch           #
#       modified by Frederic Chamot                                                           #
###############################################################################################

from EfficientNet_utils import *

import seaborn as sns
# Setting a plot style.
sns.set(style="darkgrid")


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()

        # A placeholder for metric plots.
        self.metric_plots = dict()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

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
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        # self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._fc = nn.Sequential(

            nn.Dropout(self._global_params.dropout_rate),
            nn.Linear(out_channels, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self._global_params.dropout_rate),
            nn.Linear(16, self._global_params.num_classes),
        )
        self._swish = MemoryEfficientSwish()

        """
        # tryout: freeze layers for CNN model as well
        for block in self._blocks:
            for p in block.parameters():
                p.requires_grad = False
        
        for p in self._conv_head.parameters():
            p.requires_grad = False
        """

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3, dropout_rate=0.2,
                        drop_connect_rate=0.2):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes, 'dropout_rate': dropout_rate,
                                                           'drop_connect_rate': drop_connect_rate})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class EfficientLSTM(nn.Module):
    """
    class holding the EfficientNet + LSTM model type
    """

    def __init__(self,
                 efficientnet_name='efficientnet-b3',
                 seq_len=5,
                 hidden_size=256,
                 dropout_rate=0.5,
                 drop_connect_rate=0.2,
                 recurrent_dropout_rate=0.5,
                 device=None,
                 cnn_pretrained=False,
                 cnn_checkpoint=None,
                 gradcam_mode=False):

        super().__init__()

        print(f"Initializing EfficientLSTM with {efficientnet_name} backbone")

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.device = device
        self.efficientnet_name = efficientnet_name
        self.cnn_checkpoint = cnn_checkpoint
        self.gradcam_mode = gradcam_mode

        if cnn_pretrained:
            self.effnet = EfficientNet.from_pretrained(
                                                self.efficientnet_name,
                                                num_classes=1,
                                                dropout_rate=self.dropout_rate,
                                                drop_connect_rate=self.drop_connect_rate
            )
        else:
            self.effnet = EfficientNet.from_name(
                                                self.efficientnet_name,
                                                override_params={
                                                    'num_classes': 1,
                                                    'dropout_rate': self.dropout_rate,
                                                    'drop_connect_rate': self.drop_connect_rate
                                                }
            )

        if cnn_checkpoint:
            self.effnet.load_state_dict(torch.load(self.cnn_checkpoint)['model'])
            print('Loaded own effnet checkpoint into CNN')


        # due to restrinctions, training on local machine: train only lstm and linear layers, freeze CNN layers
        #for p in self.effnet.parameters():
        #    p.requires_grad = False

        # adjust linear layer dimensions according to CNN output featurespace dimensions
        if self.efficientnet_name == 'efficientnet-b0':
            feature_dim = 62720
        elif self.efficientnet_name == 'efficientnet-b3':
            feature_dim = 75264
        else:
            assert False, f"Feature dimensions not yet implemented for model name {self.efficientnet_name}"

        # tryout: single fully connected layer between feature output of CNN and first layer of LSTM

        self.pre_linear = nn.Sequential(

            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
        )


        # LSTM
        self.rnn = nn.LSTM(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.seq_len,
                           batch_first=True,
                           dropout=self.recurrent_dropout_rate)

        # fully connected layers used for classification
        self.linear = nn.Sequential(

            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(16, 1),
            # do sigmoid in training script
        )

        # A placeholder for metric plots.
        self.metric_plots = dict()

    def forward(self, x):
        if not self.gradcam_mode:
            # get dims
            batch, seq_len, c, h, w = x.size()

            # reformat for feedforward through CNN
            cnn_input = x.view(batch * seq_len, c, h, w)
            # print(f"CNN in: {cnn_input.shape}")

            # push through conv block
            cnn_out = self.effnet.extract_features(cnn_input)
            #print(f"CNN out: {cnn_out.shape}")

            pre_linear_in = cnn_out.flatten(start_dim=1)
            #print(f"prelinear in: {pre_linear_in.shape}")

            pre_linear_out = self.pre_linear(pre_linear_in)
            #print(f"prelinear out: {pre_linear_out.shape}")

            # reformat for feed through RNN
            rnn_in = pre_linear_out.view(batch, seq_len, -1)
            #print(f"RNN in: {rnn_in.shape}")

            h0 = torch.randn(self.seq_len, rnn_in.size(0), self.hidden_size).to(self.device)
            c0 = torch.randn(self.seq_len, rnn_in.size(0), self.hidden_size).to(self.device)

            rnn_out, (h_n, c_n), = self.rnn(rnn_in, (h0, c0))
            # print(f"RNN out: {rnn_out.shape}")
            # print(h_n.shape)
            # print(h_c.shape)

            pred = self.linear(rnn_out[:, -1, :])

            return pred
        else:
            return self.effnet(x)
