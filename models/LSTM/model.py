import torch
import pickle
import os
import requests
from models.LSTM.config import MODEL_URL, SRC_URL, TGT_URL
from models.LSTM.attension import Attention
from models.LSTM.encoder import Encoder
from models.LSTM.seq2seq import Seq2Seq
from models.LSTM.seq2seq import Seq2Seq
from models.LSTM.decoder import Decoder
from models.LSTM.utils import device

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

def load_model():
    download(MODEL_URL, "LSTM/model.pth")
    download(SRC_URL, "LSTM/src_vocab.pkl")
    download(TGT_URL, "LSTM/tgt_vocab.pkl")

    
    src_vocab = pickle.load(open("LSTM/src_vocab.pkl", "rb"))
    tgt_vocab = pickle.load(open("LSTM/tgt_vocab.pkl", "rb"))
    model = Seq2Seq(
        Encoder(len(src_vocab), 256, 512),
        Decoder(len(tgt_vocab), 256, 512, Attention(512)),
        device,tgt_vocab
    )

    model.load_state_dict(torch.load("LSTM/model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab