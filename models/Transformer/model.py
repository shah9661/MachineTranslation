import torch
import pickle
import os
import requests
from models.Transformer.config import MODEL_URL, SRC_URL, TGT_URL
from models.Transformer.transfomer_ import TransformerModel
from models.Transformer.utils import device


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)



def load_model():
    download(MODEL_URL, "Transformer/model.pth")
    download(SRC_URL, "Transformer/src_vocab.pkl")
    download(TGT_URL, "Transformer/tgt_vocab.pkl")

    
    src_vocab = pickle.load(open("Transformer/src_vocab.pkl", "rb"))
    tgt_vocab = pickle.load(open("Transformer/tgt_vocab.pkl", "rb"))
    model = TransformerModel(len(src_vocab), len(tgt_vocab)).to(device)

    model.load_state_dict(torch.load("Transformer/model.pth", map_location=device))
    model.to(device)

    return model, src_vocab, tgt_vocab