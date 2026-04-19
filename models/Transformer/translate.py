from models.Transformer.utils import clean_english, encode_src, pad
import torch
from models.Transformer.config import SOS, EOS, MAX_LEN,  PAD
from models.Transformer.utils import device

def translate(sentence, model, src_vocab, tgt_vocab, device=device, max_len=MAX_LEN):
    model.eval()

    inv_vocab = {v: k for k, v in tgt_vocab.items()}

    sentence = clean_english(sentence)

    src = torch.tensor(
        pad(encode_src(sentence, src_vocab), src_vocab),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    tgt = torch.tensor([[tgt_vocab[SOS]]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):

            src_pad_mask = (src == src_vocab[PAD])
            tgt_pad_mask = (tgt == tgt_vocab[PAD])

            out = model(
                src,
                tgt,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask
            )

            next_token = out[:, -1, :].argmax(-1).item()

            tgt = torch.cat([
                tgt,
                torch.tensor([[next_token]], dtype=torch.long).to(device)
            ], dim=1)

            if next_token == tgt_vocab[EOS]:
                break

    tokens = tgt.squeeze().tolist()
    if isinstance(tokens, int):
        tokens = [tokens]

    return " ".join([
        inv_vocab[i] for i in tokens
        if i not in (tgt_vocab[SOS], tgt_vocab[EOS], tgt_vocab[PAD])
    ])