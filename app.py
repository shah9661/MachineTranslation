import streamlit as st
from utils.inference import load_model, translate

st.title("English to Hindi Translation")

model, src_vocab, tgt_vocab = load_model()
st.subheader("Input English Sentence less than 40 words")
st.info("<unk> token will be used for out-of-vocabulary words")

text = st.text_area("Enter English sentence")

if st.button("Translate"):
    if text.strip() != "":
        output = translate(text, model, src_vocab, tgt_vocab)
        
        st.subheader("Translation:")
        st.success(output)
    else:
        st.warning("Please enter input text")