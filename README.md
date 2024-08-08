# next-word-prediction
This project demonstrates how to use the GPT-2 model from Hugging Face's Transformers library to perform next word prediction. The script allows you to input a text prompt, and the model will generate a continuation of that prompt.

Requirements
Python 3.7 or higher
PyTorch
Transformers library by Hugging Face
You can install the necessary libraries using pip:
```python
pip install torch transformers
```
How It Works
The script loads a pre-trained GPT-2 model and its corresponding tokenizer. The model is then set to evaluation mode, which means it will only be used for inference, not training.
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # You can also try "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```
Generating Text
The function generate_text is defined to take a text prompt and generate a continuation of that prompt. The key parameters include:

max_length: The maximum length of the generated text.
temperature: Controls the randomness of predictions by scaling the logits before applying softmax.
```python
def generate_text(prompt, max_length=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
```
Example Usage
To generate text, simply call the generate_text function with your desired prompt:
```python
prompt = "what i am doing"
generated_text = generate_text(prompt, max_length=100)
print(generated_text)
```
## output

![Screenshot (198)](https://github.com/user-attachments/assets/614b6b37-bb91-4e6b-a925-c5a494aa0610)

Here's a conclusion you could use for your next word prediction project using the Transformers library:

## Conclusion
This project successfully demonstrates the power and versatility of the GPT-2 model in generating coherent text continuations based on a given prompt. By leveraging Hugging Face's Transformers library, we've made it easy to load and interact with pre-trained language models, enabling complex text generation tasks with just a few lines of code.
