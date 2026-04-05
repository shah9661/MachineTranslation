import torch
import pickle
import re
import os
import requests
from models.encoder import Encoder
from models.attension import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

device = torch.device("cpu")

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

MODEL_URL = "https://huggingface.co/shan9661/machineTranslation/resolve/main/model.pth"
SRC_URL = "https://huggingface.co/shan9661/machineTranslation/resolve/main/src_vocab.pkl"
TGT_URL = "https://huggingface.co/shan9661/machineTranslation/resolve/main/tgt_vocab.pkl"

def load_model():
    download(MODEL_URL, "model/model.pth")
    download(SRC_URL, "model/src_vocab.pkl")
    download(TGT_URL, "model/tgt_vocab.pkl")

    
    src_vocab = pickle.load(open("model/src_vocab.pkl", "rb"))
    tgt_vocab = pickle.load(open("model/tgt_vocab.pkl", "rb"))
    model = Seq2Seq(
        Encoder(len(src_vocab), 256, 512),
        Decoder(len(tgt_vocab), 256, 512, Attention(512)),
        device,tgt_vocab
    )

    model.load_state_dict(torch.load("model/model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab

def translate(sentence, model, src_vocab, tgt_vocab, max_len=40, device=device):
    
    model.eval()
    
    # inverse vocab (index  word)
    inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}

    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", sentence)
    tokens = sentence.split()
    
    # encode input
    src_tokens = [src_vocab.get(w, src_vocab["<unk>"]) for w in tokens]
    
    # pad
    src_tokens = src_tokens[:max_len] + [0]*(max_len - len(src_tokens))
    
    src_tensor = torch.tensor([src_tokens]).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    # start token
    input_token = torch.tensor([tgt_vocab["<sos>"]]).to(device)
    
    output_sentence = []
    
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
        
        pred_token = output.argmax(1).item()
        
        if pred_token == tgt_vocab["<eos>"]:
            break
        
        output_sentence.append(inv_tgt_vocab.get(pred_token, "<unk>"))
        
        input_token = torch.tensor([pred_token]).to(device)
    
    return " ".join(output_sentence)
