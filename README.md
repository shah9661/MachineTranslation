```
===============================================================================================
Layer (type:depth-idx)                                                 Param #
===============================================================================================
TransformerModel                                                       --
├─Embedding: 1-1                                                       17,921,024
├─Embedding: 1-2                                                       17,921,024
├─PositionalEncoding: 1-3                                              --
├─Transformer: 1-4                                                     --
│    └─TransformerEncoder: 2-1                                         --
│    │    └─ModuleList: 3-1                                            3,159,040
│    │    └─LayerNorm: 3-2                                             512
│    └─TransformerDecoder: 2-2                                         --
│    │    └─ModuleList: 3-3                                            4,213,760
│    │    └─LayerNorm: 3-4                                             512
├─Linear: 1-5                                                          17,991,028
===============================================================================================
Total params: 61,206,900
Trainable params: 61,206,900
Non-trainable params: 0
===============================================================================================
```
<img width="467" height="93" alt="image" src="https://github.com/user-attachments/assets/31d4affd-9b74-4a0c-9449-f746fb7bcdd1" />


70000 of vocab

**Model** =https://huggingface.co/shan9661/translation_transformer/tree/main

**App**= https://kbjungup5crv4upamjg9ge.streamlit.app/

