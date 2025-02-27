import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set page configuration
st.set_page_config(
    page_title="Pseudocode to C++ Translator", page_icon="🚀", layout="centered"
)

# Inject custom CSS for a sleek look
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4B4B4B;
        font-size: 36px;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #4B9CD3;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4078A0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🚀 Pseudocode to C++ Code Translator")
st.write(
    "Enter your pseudocode below, and the model will generate the corresponding C++ code!"
)


# Load model and tokenizer (cached for faster reloads)
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("pseudocode_to_cpp_model")
    tokenizer = T5Tokenizer.from_pretrained("pseudocode_to_cpp_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


model, tokenizer, device = load_model()

# Define task prefix (same as used during training)
TASK_PREFIX = "translate pseudocode to cpp: "

# Text area for pseudocode input
pseudocode_input = st.text_area("Enter Pseudocode:", height=150)

# Button to trigger translation
if st.button("Translate"):
    if pseudocode_input.strip():
        with st.spinner("Translating..."):
            # Prepare input with the task prefix
            input_text = TASK_PREFIX + pseudocode_input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            input_ids = inputs.input_ids.to(device)
            # Generate output using beam search
            outputs = model.generate(
                input_ids, max_length=128, num_beams=4, early_stopping=True
            )
            cpp_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated C++ Code:")
        st.code(cpp_code, language="cpp")
    else:
        st.error("Please enter some pseudocode before translating.")

st.markdown("Made with ❤️ using Hugging Face Transformers & Streamlit!")
