import torch
import torch.nn as nn
import math
from models.Transformer.PositionalEncoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=False
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.fc.weight = self.tgt_emb.weight

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool),
            diagonal=1
        )

        src = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        out = out.permute(1, 0, 2)
        return self.fc(out)