import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

st.title("GPT-2 Text Generator")
#st.write("Preview of dataset used:")
#st.dataframe(df.head())

# Load model + tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
st.write(f"Device set to use {device}, So text generation may be slower than usual.")

# Input prompt
suggestions = ["train delayed", "health benefits of running", "football match highlights"]
prompt = st.selectbox("Choose/Enter a prompt:", suggestions)

if st.button("Generate"):
    # ✅ Fix truncation warning: add truncation=True
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=50  # prompt length cap
    ).to(device)

    # ✅ Remove max_length in generate() to avoid conflict
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # this will control generated text length
        do_sample=True,
        temperature=0.5,
        top_p=0.85,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("### Generated Text")
    st.write(result)
