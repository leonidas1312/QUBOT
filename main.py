import streamlit as st
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)


def generate_text(prompt, history):
    # Concatenate the prompt with the history
    # full_prompt = history + prompt
    full_prompt = prompt
    # Encode the full prompt as input ids
    input_ids = tokenizer.encode(full_prompt, return_tensors='tf')

    sample_output = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
     )
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # return generated_text
    return ".".join(generated_text.split(".")[:-1]) + "."


# Define the main interface
st.title("Language Model Interface")
# ss
history = ""
prompt = st.text_input("Enter your question:")
if st.button("Submit"):
    history += prompt + " "
    answer = generate_text(prompt, history)
    st.write("Answer:", answer)
