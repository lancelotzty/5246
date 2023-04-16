import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss = nn.BCELoss(weight=targets[:,1])(data[:,0],targets[:,0])
    return bce_loss


class TransEncoder(nn.Module):

    def __init__(self, embedding_matrix, nhead=5, d_hid=1024, nlayers=10, dropout=0.5):
        super().__init__()
        self.ntoken, self.d_model = embedding_matrix.shape
        self.encoder = load_embedding_layer(embedding_matrix)
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        self.embedding_dropout = SpatialDropout(0.3)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.embedding_dropout(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # print('trans_encoder', output.shape)
        
        return output
    

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_unit):
        super(LSTMEncoder, self).__init__()
        self.embedding_size = embedding_matrix.shape[1]
        self.embedding = load_embedding_layer(embedding_matrix)
        self.embedding_dropout = SpatialDropout(0.3)
        self.lstm1 = nn.LSTM(self.embedding_size, num_unit, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(num_unit * 2, int(num_unit / 2), bidirectional=True, batch_first=True)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        h_lstm1, _ = self.lstm1(h_embedding)
        encoder_output, _ = self.lstm2(h_lstm1)
        return encoder_output


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):

        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class SpatialDropout(nn.Module):

    def __init__(self,p):
        super(SpatialDropout, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # convert to [batch, feature, timestep]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)   # back to [batch, timestep, feature]
        return x


def load_embedding_layer(embedding_matrix):
    embedding = nn.Embedding(*embedding_matrix.shape)
    embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    embedding.weight.requires_grad = False
    return embedding


class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_unit, MAX_LEN=512):
        super().__init__()
        self.lstm = LSTMEncoder(embedding_matrix, num_unit)
        self.attention = Attention(num_unit, MAX_LEN)
        self.trans = TransEncoder(embedding_matrix)
        
        self.linear1 = nn.Linear(num_unit * 2, num_unit)
        self.linear2 = nn.Linear(num_unit * 2, num_unit)
        self.linear_out = nn.Linear(num_unit, 1)
        self.linear_aux_out = nn.Linear(num_unit, 26) 

    def forward(self, x):
        # encoder_output = self.lstm(x)
        encoder_output = self.trans(x)
        # attention
        # att = self.attention(encoder_output)
        # global average pooling
        avg_pool = torch.mean(encoder_output, 1)

        # global max pooling
        max_pool, _ = torch.max(encoder_output, 1)

        # concatenation
        hidden = torch.cat((max_pool, avg_pool), 1)
        # h = torch.cat((max_pool, avg_pool, att), 1)

        h_linear1 = F.relu(self.linear1(hidden))
        h_linear2 = F.relu(self.linear2(hidden))

        out1 = F.sigmoid(self.linear_out(h_linear1))
        out2 = F.sigmoid(self.linear_aux_out(h_linear2))

        return out1, out2


# Transformer models

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NeuralNetBase(nn.Module):

    def __init__(self, num_unit, num_heads):
        super(NeuralNetBase, self).__init__()
        # self.attention = Attention(num_unit, MAX_LEN)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, num_unit)
        self.linear_out = nn.Linear(num_unit, 1)
        
    def forward(self, x_context_embedding):

        # x_context_embedding = torch.unsqueeze(x_context_embedding, 1)
        h_linear1 = F.dropout(F.relu(self.linear1(x_context_embedding)), 0.4)
        h_linear2 = F.dropout(F.relu(self.linear2(h_linear1)), 0.4)

        out1 = torch.sigmoid(self.linear_out(h_linear2))

        return out1