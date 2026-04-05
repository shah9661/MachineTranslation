import torch
import torch.nn as nn
# Attention
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs): # will pass layer hidden vector
        seq_len = encoder_outputs.shape[1] # seq_len=sentance lentgh after padding 
        # hidden 1 batch_size dim
        hidden = hidden.repeat(seq_len,1,1).permute(1,0,2) # repeat seq_len time and change dimension
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) # concat dim base dim 2
        attention = self.v(energy).squeeze(2) # drop dim=2

        return torch.softmax(attention, dim=1) # return batch sizr and seq_len pro