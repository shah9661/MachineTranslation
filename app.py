import streamlit as st
from models.LSTM.model import load_model as load_lstm_model
from models.LSTM.translate import translate as lstm_translate
from models.Transformer.model import load_model as load_transformer_model
from models.Transformer.translate import translate as transformer_translate

@st.cache_resource
def load_lstm():
    return load_lstm_model()

@st.cache_resource
def load_transformer():
    return load_transformer_model()


st.sidebar.title("Select Model")
model_name = st.sidebar.selectbox("Choose model", ["LSTM", "Transformer"])


st.title("English to Hindi Translation")

st.subheader("Input English Sentence (max 40 words)")
st.info("<unk> token will be used for out-of-vocabulary words")

text = st.text_area("Enter English sentence")


if model_name == "LSTM":
    model_obj, src_vocab, tgt_vocab = load_lstm()
    translate_func = lstm_translate

else:
    model_obj, src_vocab, tgt_vocab = load_transformer()
    translate_func = transformer_translate
    
if st.button("Translate"):
    if text.strip() !="":
        output = translate_func(
            text,
            model_obj,
            src_vocab,
            tgt_vocab,
            device="cpu", 
            max_len=40
        )

        st.subheader("Translation:")
        st.success(output)
        
    else:
        st.warning("Please enter input text")