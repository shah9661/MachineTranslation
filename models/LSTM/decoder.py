import torch
import torch.nn as nn
# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(hid_dim+emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim) # hid_to to vocab_size
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1) # adding one dim input = 3 dim
        embedded = self.embedding(input) # vector 

        attn = self.attention(hidden[-1], encoder_outputs) # hidden[-1] last LSTM layer hidden state one step at a time
        attn = attn.unsqueeze(1) # adding one dim

        context = torch.bmm(attn, encoder_outputs) # batch multi

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell)) #batch_size, 1, hid_dim

        prediction = self.fc(output.squeeze(1)) 
        return prediction, hidden, cell