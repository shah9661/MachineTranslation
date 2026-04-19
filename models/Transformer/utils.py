import re
import torch
from models.Transformer.config import PAD,  EOS, UNK, MAX_LEN
device = torch.device("cpu")
def clean_english(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def pad(seq, vocab):
    pad_idx = vocab[PAD]
    return seq[:MAX_LEN] + [pad_idx] * (MAX_LEN - len(seq))

def encode_src(sentence, vocab):
    tokens = sentence.split()
    return [vocab.get(w, vocab[UNK]) for w in tokens] + [vocab[EOS]]