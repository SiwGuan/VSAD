import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
import math
import dgl
from dgl.nn.pytorch import GATConv,GatedGraphConv
import numpy as np

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp1 = Chomp1d(padding)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout2d(dropout)

    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation))
    self.chomp2 = Chomp1d(padding)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout2d(dropout)

    self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                             self.conv2, self.chomp2, self.relu2, self.dropout2)
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
    nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
    if self.downsample is not None:
        nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

  def forward(self, x):
    net = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(net + res)


class TemporalConvNet(nn.Module):
  def __init__(self, num_inputs, num_channels,d_model,kernel_size=2, dropout=0.2, attention=True):
    super(TemporalConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size-1) * dilation_size, dropout=dropout)]
      if attention == True:
        layers += [AttentionBlock(num_channels[-1],d_model)]

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)

class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, hidden,d_model):
    super(AttentionBlock, self).__init__()
    self.emb = nn.Linear(hidden,d_model)
    self.key_layer = nn.Linear(d_model, d_model)
    self.query_layer = nn.Linear(d_model, d_model)
    self.value_layer = nn.Linear(d_model, d_model)
    self.sqrt_k = math.sqrt(d_model)

  def forward(self, minibatch):
    minibatch = self.emb(minibatch.permute(0,2,1))
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = torch.tril(torch.ones(logits.size())).to(torch.bool).cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    inf = torch.tensor(-2**15+1.0,dtype=torch.float32).cuda()
    logits = torch.where(mask,logits,inf)
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    output = read + minibatch
    return output.permute(0,2,1)



class Transform(nn.Module):
    def __init__(self, d_model, head,device):
        super(Transform, self).__init__()
        self.qff = nn.Linear(d_model, d_model)
        self.kff = nn.Linear(d_model, d_model)
        self.vff = nn.Linear(d_model, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.lnff = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.d_model =d_model
        self.head = head
        self.mask = True
        self.device = device
    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)
        A = torch.matmul(query, key.transpose(2,1))
        if self.mask:
            batch_size = x.shape[0]
            window = x.shape[1]
            mask = torch.ones(window,window).to(self.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(mask,0)
            mask = mask.repeat(batch_size,1,1)
            mask = mask.to(torch.bool)
            inf = torch.tensor(-2 ** 15 + 1.0,dtype=torch.float32).to(self.device)
            A = torch.where(mask,A,inf)
        A /= (self.d_model ** 0.5)
        A = torch.softmax(A, -1)
        value = torch.matmul(A, value)
        value += x
        value = self.ln(value)
        x = self.lnff(self.ff(value))
        return x


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=200, dropout=0.1):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
    pe += torch.sin(position * div_term)
    pe += torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x, pos=0):
    x = x + Variable(self.pe[pos:pos + x.size(0), :],requires_grad=False)
    return self.dropout(x)



class SGNN(nn.Module):
    def __init__(self, outfea):
        super(SGNN, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),nn.ReLU(),
            nn.Linear(outfea, outfea),nn.Sigmoid(),
        )
        self.ff1 = nn.Linear(outfea, outfea)
        self.ff2 = nn.Sequential(nn.Linear(outfea, outfea),nn.ReLU(),
                                 nn.Linear(outfea, outfea),nn.ReLU(),)
    def forward(self, x):
        p = self.ff(x)
        a = torch.matmul(p, p.transpose(-1, -2))
        R = torch.relu(torch.softmax(a, -1)) + torch.eye(x.shape[1]).cuda()

        D = (R.sum(-1) ** -0.5)
        D[torch.isinf(D)] = 0.
        D = torch.diag_embed(D)

        A = torch.matmul(torch.matmul(D, R), D)
        x = torch.relu(self.ff1(torch.matmul(A, x)))
        return x

class predictModule(nn.Module):
    def __init__(self, d_model,n_feature):
        super(predictModule,self).__init__()


        self.linear = nn.Sequential(
            nn.Linear(d_model,n_feature),nn.ReLU(),
            nn.Linear(n_feature, n_feature), nn.ReLU(),
            nn.Linear(n_feature, n_feature), nn.ReLU(),
            nn.Linear(n_feature, 1), nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.linear(x)
        return x
