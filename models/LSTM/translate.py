import torch
from models.LSTM.config import  MAX_LEN, SOS, EOS
from models.LSTM.utils import preprocess, encode, device


def translate(sentence, model, src_vocab, tgt_vocab, device=device, max_len=MAX_LEN):
    model.eval()

    # inverse vocab (index → word)
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    # preprocess + encode
    tokens = preprocess(sentence)
    src_tensor = encode(tokens, max_len, src_vocab, device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # start token
    input_token = torch.tensor(
        [tgt_vocab[SOS]],
        device=device
    )

    output_sentence = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )

        pred_token = output.argmax(1).item()

        if pred_token == tgt_vocab[EOS]:
            break

        word = inv_tgt_vocab.get(pred_token, "<unk>")
        output_sentence.append(word)

        input_token = torch.tensor([pred_token], device=device)

    return " ".join(output_sentence)