# English-to-Hindi Neural Machine Translation (Transformer from Scratch)

A from-scratch PyTorch implementation of the original Transformer architecture (Vaswani et al., *Attention Is All You Need*) for English → Hindi translation, trained on a large-scale parallel corpus using dual T4 GPUs on Kaggle.

---

## 🚀 Highlights

- **Pure PyTorch implementation** — built on top of `nn.Transformer`, with custom embeddings, positional encoding, masking, and greedy decoding.
- **61.2M parameters** — encoder-decoder Transformer (`d_model=256`, `nhead=8`, 4 encoder + 4 decoder layers).
- **Multi-GPU training** — `nn.DataParallel` across dual T4 GPUs with mixed-precision (AMP) training.
- **Incremental / streaming training strategy** — custom memory-replay sampling to train on a huge parallel corpus in chunks without catastrophic forgetting.
- **Checkpointing & resume support** — model, optimizer, scaler, and replay memory are all persisted and restorable.
- **Evaluation suite** — token-level accuracy and BLEU score (via NLTK) on held-out samples.

---

## 🏗️ Architecture

| Component | Detail |
|---|---|
| Model type | Encoder-Decoder Transformer (`torch.nn.Transformer`) |
| `d_model` | 256 |
| Attention heads | 8 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Feedforward dim | 1024 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal (fixed, non-learned) |
| Weight tying | Decoder output projection tied to target embedding weights |
| Total parameters | **61,206,900** |

```
Total params: 61,206,900
Trainable params: 61,206,900
Non-trainable params: 0
```

---

## 📚 Dataset & Preprocessing

- Parallel English-Hindi sentence corpus (line-aligned `.txt` files).
- **English cleaning:** lowercasing, URL removal, non-alphanumeric stripping (retaining basic punctuation), whitespace normalization.
- **Hindi cleaning:** removal of zero-width/invisible Unicode characters, restriction to Devanagari script range (`\u0900–\u097F`) plus basic punctuation, whitespace normalization.
- **Vocabulary:** built independently for source (English) and target (Hindi), capped at 70,000 tokens each, with special tokens `<pad>`, `<sos>`, `<eos>`, `<unk>`.
- **Corpus scale:** ~984K unique English tokens and ~1.19M unique Hindi tokens observed across the full corpus.
- Sequences are truncated/padded to a fixed `MAX_LEN = 40`.

---

## 🔁 Training Strategy

Training a Transformer on a multi-million-sentence corpus within Kaggle's session limits required an **incremental training pipeline** instead of loading the full dataset at once:

1. **Chunked exposure:** The corpus is split into large sub-chunks (e.g. 1M sentence pairs), from which random subsets are sampled per training run.
2. **Memory replay:** Each epoch's training batch is composed of ~30% samples from a rolling "memory" buffer (previously seen data) and ~70% fresh samples from the current chunk — this mitigates catastrophic forgetting across incremental training sessions.
3. **Resumable checkpoints:** Every epoch saves model weights, optimizer state, AMP scaler state, and the current replay memory to `checkpoint.pth`, allowing training to be resumed exactly where it left off (across Kaggle sessions/GPU quota resets).
4. **Mixed precision training:** `torch.amp.autocast` + `GradScaler` for faster training and lower memory footprint on T4 GPUs.
5. **Gradient clipping:** `clip_grad_norm_` at 1.0 to stabilize training.

```
model = train(5, pairs, src_vocab, tgt_vocab, device, resume=False)   # initial training
model = train(10, sub_pairs, src_vocab, tgt_vocab, device, resume=True)  # incremental fine-tuning on new chunks
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Token-level accuracy | **37.82%** |
| BLEU score | Computed via NLTK `sentence_bleu` with smoothing (method4) |

> Note: These results reflect an early-stage, compute-constrained training run (limited epochs and sampled sub-chunks of a much larger corpus). Performance is expected to improve substantially with longer training, larger sampled chunks, and beam search decoding instead of greedy decoding.

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── en.txt                # raw English sentences
│   └── hi.txt                # raw Hindi sentences (line-aligned)
├── vocab/
│   ├── src_vocab.pkl
│   └── tgt_vocab.pkl
├── checkpoint.pth            # model + optimizer + scaler + replay memory
├── model.py                  # TransformerModel, PositionalEncoding
├── dataset.py                # TranslationDataset, encode/pad utilities
├── train.py                  # training loop, incremental training driver
├── evaluate.py                # accuracy + BLEU evaluation
└── README.md
```

---

## ⚙️ Setup

```bash
pip install torch numpy pandas nltk
```

### Training from scratch
```python
src_vocab = build_vocab([p[0] for p in pairs])
tgt_vocab = build_vocab([p[1] for p in pairs])
model = train(epochs=5, sub_pairs=pairs, src_vocab=src_vocab, tgt_vocab=tgt_vocab, device=device, resume=False)
```

### Resuming / incremental training
```python
model = train(epochs=10, sub_pairs=new_chunk, src_vocab=src_vocab, tgt_vocab=tgt_vocab, device=device, resume=True)
```

### Inference
```python
model = load_model_for_inference(src_vocab, tgt_vocab, device)
```

### Evaluation
```python
accuracy = evaluate(model, loader, device, src_vocab, tgt_vocab)
bleu = compute_bleu(model, loader, src_vocab, tgt_vocab, device)
```

---

## 🔍 Known Limitations

- Greedy decoding is used at inference — no beam search yet, which caps translation quality.
- Vocabulary is word-level (not subword/BPE), so out-of-vocabulary words are mapped to `<unk>`, hurting rare-word translation.
- Training exposure per run is a sampled subset of the full corpus due to Kaggle compute/time constraints.
- Fixed `MAX_LEN = 40` truncates longer sentences.

## 🛣️ Future Work

- [ ] Switch to subword tokenization (BPE / SentencePiece) to shrink vocab and handle OOV words.
- [ ] Add beam search decoding for inference.
- [ ] Full-corpus training with a proper train/val/test split and early stopping.
- [ ] Track BLEU/accuracy curves across incremental training stages.
- [ ] Package as a simple inference API / demo app.

---

## 🙏 Acknowledgments

- Vaswani et al., *"Attention Is All You Need"* (2017) — Transformer architecture.
- Trained on Kaggle using dual NVIDIA T4 GPUs.

## 📄 License

MIT License — feel free to use, modify, and build upon this work with attribution.
