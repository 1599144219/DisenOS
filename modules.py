import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(Attention, self).__init__()
        self.fc11 = nn.Linear(d_in, 250)
        self.fc22 = nn.Linear(250, d_out)
        init.xavier_normal_(self.fc11.weight)
        init.xavier_normal_(self.fc22.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc22(self.relu(self.fc11(x)))

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        '''
        d_hid: 2*embed_dim
        '''
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class SupportEncoder(nn.Module):
    """
    两层全连接: 对实体对进行再次编码
    """

    def __init__(self, d_model, d_inner, dropout=0.1):
        '''
        d_model: 2*embed_dim
        d_inner: 2*d_model
        '''
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x: [batch/few, 2*embed_dim]
        '''
        residual = x
        # [batch/few, 4*embed_dim]
        output = self.relu(self.proj1(x))
        # [batch/few, 2*embed_dim]
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)


class QueryEncoder(nn.Module):
    """
    LSTM: 对query中的实体对进行再次编码
    """

    def __init__(self, input_dim, process_step=4):
        '''
        input_dim: 2*embed_dim
        '''
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        # 第一个参数feature_len: 2*embed_dim
        # 第二个参数hidden_layer: 4*embed_dim
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support, query):
        '''
        support: (few, 2*embed_dim)
        query: (batch_size, 2*embed_dim)
        return: (batch_size, 2*embed_dim)
        '''
        if self.process_step == 0:
            return query
        batch_size = query.size()[0]
        # 隐藏单元和记忆单元 [batch, 4*embed_dim]
        h_r = torch.zeros(batch_size, 2 * self.input_dim).cuda()
        c = torch.zeros(batch_size, 2 * self.input_dim).cuda()
        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            # [batch_size, 2*embed_dim]
            h = query + h_r_[:, :self.input_dim]
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            # [batch_size, 2*embed_dim]
            r = torch.matmul(attn, support)
            # [batch_size, 4*embed_dim]
            h_r = torch.cat((h, r), dim=1)

        return h
