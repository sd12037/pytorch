import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.modules.rnn as RNNBase
from torch.nn.modules.rnn import LSTMCell

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

class Wavelet_cnn(nn.Module):
    '''
    input size is (batch, 1, seq, electrode)
    seq is expected 2^n
    output[0]: decomposition: size is (batch, 1, seq, electrode)
    output[1]: scalegram: size is (batch, level, seq, electrode)
    output[2]: list of pure coef
    '''
    def __init__(self, conv_length=32, stride=2, level=100):
        super(Wavelet_cnn, self).__init__()
        pad = conv_length - 1
        self.pad = nn.ZeroPad2d((0, 0, pad, 0))
        self.conv_length = conv_length
        self.stride = stride
        self.level = level
        ### l_pass, h_pass
        self.l_conv = nn.Conv2d(1, 1, (conv_length, 1),
                                stride=(1, 1), bias=False)
        self.h_conv = nn.Conv2d(1, 1, (conv_length, 1),
                                stride=(1, 1), bias=False)
        ### down sampler
        self.l_downsample = nn.Conv2d(1, 1, (1, 1),
                                      stride=(stride, 1), bias=False)
        self.h_downsample = nn.Conv2d(1, 1, (1, 1),
                                      stride=(stride, 1), bias=False)
        ### up sampler
        self.upsample = nn.ConvTranspose2d(1, 1, (stride, 1),
                                        stride=(stride, 1), bias=False)
        self.init_filter()


    def init_filter(self):
        self.l_conv.weight.data.fill_(1.0*(1/self.l_conv.weight.size(2)))
        self.h_conv.weight.data.fill_(1.0*(1/self.h_conv.weight.size(2)))
        for i in range(0, self.conv_length, 2):
            self.h_conv.weight.data[:,:,i,:] *= -1

    def forward(self, x):
#         x = x.unsqueeze(1).unsqueeze(-1)
        b = x
        a_list = []
        a_coef = []
        for i in range(self.level):
            num_elec = b.size(3)
            seq_half = int(b.size(2)/2)
            # a = F.adaptive_avg_pool2d(self.h_conv(self.pad(b)),
            #                           (seq_half, num_elec))
            # b = F.adaptive_avg_pool2d(self.l_conv(self.pad(b)),
            #                           (seq_half, num_elec))
            a = self.h_downsample(self.h_conv(self.pad(b)))
            b = self.l_downsample(self.l_conv(self.pad(b)))
            a_coef.append(a)
#             print(a.size())
            for j in range(i+1):
                a = self.upsample(a)
#             print(a.size())
            a_list.append(a)

            if b.size(2) < self.stride:
                break
#         print('composition level is {}'.format(i+1))
        scalegram = torch.stack(a_list, dim=2)
        decomposition = scalegram.sum(2)

        return decomposition, scalegram, a_coef

class Wavelet_cnn(nn.Module):
    '''
    input size is (batch, 1, seq, electrode)
    seq is expected 2^n
    output[0]: decomposition: size is (batch, 1, seq, electrode)
    output[1]: scalegram: size is (batch, level, seq, electrode)
    output[2]: list of pure coef
    '''
    def __init__(self, conv_length=32, stride=2, level=100):
        super(Wavelet_cnn, self).__init__()
        pad = conv_length - 1
        self.pad = nn.ZeroPad2d((0, 0, pad, 0))
        self.conv_length = conv_length
        self.stride = stride
        self.level = level
        ### l_pass, h_pass
        self.l_conv = nn.Conv2d(1, 1, (conv_length, 1),
                                stride=(1, 1), bias=False)
        self.h_conv = nn.Conv2d(1, 1, (conv_length, 1),
                                stride=(1, 1), bias=False)
        ### down sampler
        self.l_downsample = nn.Conv2d(1, 1, (1, 1),
                                      stride=(stride, 1), bias=False)
        self.h_downsample = nn.Conv2d(1, 1, (1, 1),
                                      stride=(stride, 1), bias=False)
        ### up sampler
        self.upsample = nn.ConvTranspose2d(1, 1, (stride, 1),
                                        stride=(stride, 1), bias=False)
        self.init_filter()


    def init_filter(self):
        self.l_conv.weight.data.fill_(1.0*(1/self.l_conv.weight.size(2)))
        self.h_conv.weight.data.fill_(1.0*(1/self.h_conv.weight.size(2)))
        for i in range(0, self.conv_length, 2):
            self.h_conv.weight.data[:,:,i,:] *= -1

    def forward(self, x):
#         x = x.unsqueeze(1).unsqueeze(-1)
        b = x
        a_list = []
        a_coef = []
        for i in range(self.level):
            num_elec = b.size(3)
            seq_half = int(b.size(2)/2)
            # a = F.adaptive_avg_pool2d(self.h_conv(self.pad(b)),
            #                           (seq_half, num_elec))
            # b = F.adaptive_avg_pool2d(self.l_conv(self.pad(b)),
            #                           (seq_half, num_elec))
            a = self.h_downsample(self.h_conv(self.pad(b)))
            b = self.l_downsample(self.l_conv(self.pad(b)))
            a_coef.append(a)
#             print(a.size())
            for j in range(i+1):
                a = self.upsample(a)
#             print(a.size())
            a_list.append(a)

            if b.size(2) < self.stride:
                break
#         print('composition level is {}'.format(i+1))
        scalegram = torch.stack(a_list, dim=2)
        decomposition = scalegram.sum(2)

        return decomposition, scalegram, a_coef


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)

    def forward(self, input, h, c):

        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A,
                                       int(A.size()[1] / self.num_features),
                                       dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c,
                                     shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c,
                                     shape[0], shape[1])).cuda())

class SeqConvLSTM(nn.Module):
  def __init__(self, input_channels, hidden_channels,
               kernel_size, bias=True):
    super(SeqConvLSTM, self).__init__()
    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.bias = bias
    self.cell = ConvLSTMCell(self.input_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             self.bias)

  def forward(self, input):
    '''
    input is expected (batch_size, seq_len, ch, height, width)
    '''
    batch_size, seq_len, _, height, width = input.size()
    outputs = []
    for i in range(seq_len):
      if i == 0:
        h, c = self.cell.init_hidden(batch_size,
                                     self.hidden_channels,
                                     (height, width))
      h, c = self.cell(input[:,i,:,:,:], h, c)
      outputs.append(h)
    return torch.stack(outputs, dim=1)

class NlayersSeqConvLSTM(nn.Module):
  '''
  NlayersSeqConvLSTM
  input is Variable whose size of (batch_size, seq_len, ch, height, width)
  return[0] is Variable whose size of (batch_size, seq_len, ch, height, width)
  return[1] is list whose all Variables of hidden layers

  '''
  def __init__(self, input_channels, hidden_channels,
               kernel_sizes, bias=True):
    super(NlayersSeqConvLSTM, self).__init__()
    self.input_channels = [input_channels] + hidden_channels
    self.hidden_channels = hidden_channels
    self.kernel_sizes = kernel_sizes
    self.num_layers = len(hidden_channels)
    self.bias = bias
    self._all_layers=[]
    for i in range(self.num_layers):
      cell = SeqConvLSTM(self.input_channels[i],
                         self.hidden_channels[i],
                         self.kernel_sizes[i],
                         self.bias)
      self._all_layers.append(cell)
    self.cells = nn.ModuleList(self._all_layers)

  def forward(self, input):
    outputs = []
    h = input
    for layer in self.cells:
      h = layer(h)
      outputs.append(h)
    return h, outputs


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                                self.hidden_channels[i],
                                self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = ConvLSTMCell.init_hidden(bsize,
                              self.hidden_channels[i], (height, width))
                    internal_state.append((h, c))
                # do forward
                name = 'cell{}'.format(i)
                (h, c) = internal_state[i]

                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class Bayes_classifier(nn.Module):
  def __init__(self, predictor, num_sample, num_class=2):
    '''
    ドロップアウトネットワークのベイズ予測
    '''
    super(Bayes_classifier, self).__init__()
    self.num_sample = num_sample
    self.predictor = predictor
    self.num_class = num_class

  def forward(self, x):
    num_batch = x.size(0)
    num_class = self.num_class

    '''
    categorycal分布のパラメータw~p(w|D)をサンプリング
    そのwを用いてC~p(C|x,w)p(w|D)をサンプリング
    この操作を複数回行い、(1/N)sum_i p(C|x,w_i)p(w_i|D)
    で事後分布p(C|x,D)を近似する
    http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf(p51~52)
    '''
    bayes_predict = torch.zeros(num_batch, num_class).cuda()
    for _ in range(self.num_sample):
      _, label = torch.max(self.predictor(x), 1)
      label = label.resize(num_batch, 1)
      label_onehot = torch.zeros(num_batch, num_class).cuda()
      bayes_predict += label_onehot.scatter_(1, label.data, 1)
    bayes_predict /= self.num_sample
    return Variable(bayes_predict)


class BayesLSTM(nn.Module):
  def __init__(self, in_size, hidden_size, batch_size,
               in_dropout=0.5, hidden_dropout=0.5, out_dropout=0.5,
               bias=True, gpu=True):
    super(BayesLSTM, self).__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.cell = LSTMCell(in_size, hidden_size , bias=bias)
    self.hidden_dropout = hidden_dropout
    self.in_dropout = in_dropout
    self.out_dropout = out_dropout
    self.gpu = gpu

  def reset_state(self, x):
    self.batch_size = x.size(0)
    self.h0 = Variable(torch.zeros((self.batch_size, self.hidden_size)))
    self.c0 = Variable(torch.zeros((self.batch_size, self.hidden_size)))
    if self.gpu:
      self.h0 = self.h0.cuda()
      self.c0 = self.c0.cuda()

  def set_drop_out(self):
    self.in_drop_mask = Variable(
          torch.bernoulli(
          torch.ones(self.batch_size, self.in_size) * (1 - self.in_dropout)))
    self.hidden_drop_mask = Variable(
          torch.bernoulli(
          torch.ones(self.batch_size, self.hidden_size) * (1 - self.hidden_dropout)))
    self.out_drop_mask = Variable(
          torch.bernoulli(
          torch.ones(self.batch_size, self.hidden_size) * (1 - self.out_dropout)))

    if self.gpu:
      self.in_drop_mask = self.in_drop_mask.cuda()
      self.hidden_drop_mask = self.hidden_drop_mask.cuda()
      self.out_drop_mask = self.out_drop_mask.cuda()

  def forward(self, x):
    self.reset_state(x)
    self.set_drop_out()
    y_seq = Variable(torch.zeros((self.batch_size,
                                  x.size(1),
                                  self.hidden_size)))
    if self.gpu:
      y_seq = y_seq.cuda()
    # h_seq = Variable(torch.zeros((self.batch_size,
    #                               x.size(1),
    #                               self.hidden_size)))

    h, c = self.cell(x[:, 0, :] * self.in_drop_mask, (self.h0, self.c0))
    y = h * self.out_drop_mask
    h = h * self.hidden_drop_mask
    # h_seq[:, 0, :] = h
    y_seq[:, 0, :] = y
    for i in range(x.size(1) - 1):
      h, c = self.cell(x[:, i + 1, :], (h, c))
      y = h * self.out_drop_mask
      h = h * self.hidden_drop_mask
      # h_seq[:, i + 1, :] = h
      y_seq[:, i + 1, :] = y
    return y_seq, (h, c)



class Residual_block(nn.Module):
  '''
  https://arxiv.org/pdf/1603.05027.pdf
  で検証されたResidual Unitsの1D-ver
  '''
  def __init__(self, in_ch, kernel_size, stride, padding, dropout, bayes=False):
    super(Residual_block, self).__init__()
    self.bayes = bayes
    self.bn1 = nn.BatchNorm1d(in_ch)
    self.bn2 = nn.BatchNorm1d(in_ch)
    self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                           kernel_size=kernel_size, padding=padding,
                           stride=stride, bias=False)
    self.conv2 = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                           kernel_size=kernel_size, padding=padding,
                           stride=stride, bias=False)
    self.do = nn.Dropout(dropout)
    self.dropout_rate = dropout

  def forward(self, x):
    res = x

    h = self.bn1(x)
    h = nn.functional.relu(h)
    h = self.conv1(h)
    h = self.bn2(h)
    h = nn.functional.relu(h)
    if self.bayes:
      h = nn.functional.dropout(h, p=self.dropout_rate, training=True)
    else:
      h = self.do(h)
    h = self.conv2(h)

    return h + x


class Res_net(nn.Module):
  def __init__(self, dropout, res_ch=[10, 64, 128],
               adaptive_seq_len=[64, 32, 16],
               kernel_size=7, stride=1, padding=3,
               bayes=False):
    super(Res_net, self).__init__()

    assert len(res_ch)==len(adaptive_seq_len), 'expected len(res_ch)== \
      len(adaptive_seq_len), but actual len(res_ch)={},\
      and len(adaptive_seq_len)={}'.format(len(res_ch), len(adaptive_seq_len))

    module_list = []
    #### stacked resnet except last layers
    for i in range(len(res_ch)-1):
      module_list.append(Residual_block(in_ch=res_ch[i],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dropout=dropout,
                                        bayes=bayes))
      module_list.append(nn.AdaptiveAvgPool1d(adaptive_seq_len[i]))
      module_list.append(nn.Conv1d(in_channels=res_ch[i],
                                   out_channels=res_ch[i+1],
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   bias=False))
    #### append last layer
    module_list.append(Residual_block(in_ch=res_ch[-1],
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dropout=dropout,
                                      bayes=bayes))
    module_list.append(nn.AdaptiveAvgPool1d(adaptive_seq_len[-1]))

    self.res_block = nn.ModuleList(module_list)

  def forward(self, x):
    for f in self.res_block:
      x = f(x)
    h = x.transpose(1,2)  ## batch x ch x seq_len -> batch x seq_len x ch
    return h




class LSTM(nn.Module):
  def __init__(self, in_size, hidden_size, batch_size,
               num_layers=1, dropout=.1,
               bidirectional=False, return_seq=True,
               batch_first=True, gpu=False,
               continue_seq=False):
    super(LSTM, self).__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.batch_size = batch_size
    self.gpu = gpu
    self.bidirectional = bidirectional
    self.batch_first = batch_first
    self.return_seq = return_seq
    self.continue_seq = continue_seq
    if self.bidirectional:
      self.multi = 2
    else:
      self.multi = 1

    self.lstm = nn.LSTM(input_size=self.in_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional,
                        batch_first=self.batch_first,
                        dropout=self.dropout)

    if self.gpu:
      self.h0 = Variable(torch.zeros(self.multi*self.num_layers,
                                     self.batch_size,
                                     self.hidden_size).cuda())
      self.c0 = Variable(torch.zeros(self.multi*self.num_layers,
                                     self.batch_size,
                                     self.hidden_size).cuda())
    else:
      self.h0 = Variable(torch.zeros(2*self.num_layers,
                                     self.batch_size,
                                     self.hidden_size))
      self.c0 = Variable(torch.zeros(2*self.num_layers,
                                     self.batch_size,
                                     self.hidden_size))

  def reset_state(self, x):
    batch_size = x.size(0)
    if self.gpu:
      self.h0 = Variable(torch.zeros(self.multi*self.num_layers,
                                     batch_size,
                                     self.hidden_size).cuda())
      self.c0 = Variable(torch.zeros(self.multi*self.num_layers,
                                     batch_size,
                                     self.hidden_size).cuda())
    else:
      self.h0 = Variable(torch.zeros(2*self.num_layers,
                                     batch_size,
                                     self.hidden_size))
      self.c0 = Variable(torch.zeros(2*self.num_layers,
                                     batch_size,
                                     self.hidden_size))

  def forward(self, x):
    if self.continue_seq:
      h, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
    else:
      self.reset_state(x)
      h, _ = self.lstm(x, (self.h0, self.c0))


    if self.return_seq:
      pass
    else:
      h =  h[:,-1,:]
    return h


class T_CNN(nn.Module):
  def __init__(self, in_channels=5, out_channels=16,
               stride=1, kernel_size=3, padding=1,
               dilation=1, bias=False, dropout=.1,
               layers=1, activation=F.relu, b_n=True,
               channel_last=True):
    super(T_CNN, self).__init__()

    '''
    channnel_last = True mean input and output shape is (batch, seq, ch)
    b_n affects only last layer
    '''
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.bias = bias
    self.dropout = dropout
    self.layers = layers
    self.activation = activation
    self.b_n = b_n
    self.channel_last = channel_last

    self.set()

  def set(self):
    if self.b_n:
      self.b_n = torch.nn.BatchNorm1d(self.out_channels)

    conv_list = [torch.nn.Conv1d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 padding=self.padding,
                                 dilation=self.dilation,
                                 bias=self.bias)]

    for i in range(1, self.layers):
      conv_list.append(torch.nn.Conv1d(in_channels=self.out_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       dilation=self.dilation,
                                       bias=self.bias))

    self.conv = nn.ModuleList(conv_list)

  def forward(self, x):
    ## transpose from (time x ch) to (ch x time)
    if self.channel_last:
      h = x.transpose(1, 2)
    else:
      h = x

    h = h

    ## first layer and hidden layers
    for i in range(self.layers-1):
      h = self.activation(self.conv[i](h))
      h = F.dropout(h, p=self.dropout)

    ## last layer
    h = self.conv[-1](h)
    h = self.b_n(h)
    h = F.dropout(h, p=self.dropout)

    ## transpose from (ch x time) to (time x ch)
    if self.channel_last:
      h = h.transpose(1, 2)
    return h


# class SeqGenerator(nn.Module):
#   def __init__():
#     super(SeqGenerator, self).__init__()
#     self.rnn = nn.LSTM()
#
# class SeqDiscriminator(nn.Module)
