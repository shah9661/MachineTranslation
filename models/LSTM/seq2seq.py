import torch
import torch.nn as nn
# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device,tgt_vocab):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.tgt_vocab = tgt_vocab

    def forward(self, src, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = len(self.tgt_vocab)

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device) # empty containar

        enc_outputs, hidden, cell = self.encoder(src)
        input = trg[:,0] # batch_size and first token<sos> here 2 dim but need three dim

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, enc_outputs)
            outputs[:,t] = output # all batch fixed time step
            input = trg[:,t]

        return outputs