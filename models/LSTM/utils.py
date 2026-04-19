import re
import torch
from models.LSTM.config import UNK
device = torch.device("cpu")
def preprocess(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
        tokens = sentence.split()
        return tokens

def encode(tokens,max_len,src_vocab,device):
        src_tokens = [src_vocab.get(w, src_vocab[UNK]) for w in tokens]

        # pad
        src_tokens = src_tokens[:max_len] + [0]*(max_len - len(src_tokens))

        return torch.tensor([src_tokens]).to(device)