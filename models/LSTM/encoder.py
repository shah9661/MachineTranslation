import torch.nn as nn
# Encoder 
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src) #batch size seq_len and emb_dim
        outputs, (hidden, cell) = self.lstm(embedded) # output -->batch_size, seq_len,hid_dim
        # (hidden, cell)---> 1,batch_size,hid_dim
        return outputs, hidden, cell