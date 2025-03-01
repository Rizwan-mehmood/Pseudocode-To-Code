import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(
    page_title="Pseudocode to C++ Translator", page_icon="🚀", layout="centered"
)

st.markdown(
    """
    <style>
    body { background-color: #f0f2f6; }
    .main { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #4B4B4B; font-size: 36px; font-weight: 600; }
    .stButton>button { background-color: #4B9CD3; color: white; font-size: 18px; padding: 10px 24px; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #4078A0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Pseudocode to C++ Code Translator")
st.write(
    "Enter your pseudocode below, and the model will generate the corresponding C++ code!"
)


# Load model and tokenizer (cached for faster reloads)
@st.cache_resource
def load_model():
    # Use the same model directory as trained in Colab
    model = T5ForConditionalGeneration.from_pretrained("advanced_pseudocode_to_cpp")
    tokenizer = T5Tokenizer.from_pretrained("advanced_pseudocode_to_cpp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


model, tokenizer, device = load_model()

# Use the same task prefix and inference parameters as during training
TASK_PREFIX = "Convert the following pseudocode to optimized C++:\n"
END_TOKEN = "</>"


def convert_pseudocode(pseudocode):
    input_text = TASK_PREFIX + pseudocode + END_TOKEN
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=256,
        num_beams=8,
        do_sample=True,  # If you want to match Colab's behavior with sampling
        early_stopping=True,
        repetition_penalty=2.5,
        length_penalty=1.2,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


pseudocode_input = st.text_area("Enter Pseudocode:", height=150)

if st.button("Translate"):
    if pseudocode_input.strip():
        with st.spinner("Translating..."):
            cpp_code = convert_pseudocode(pseudocode_input)
        st.subheader("Generated C++ Code:")
        st.code(cpp_code, language="cpp")
    else:
        st.error("Please enter some pseudocode before translating.")
