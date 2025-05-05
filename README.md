# GPT-2 Micro

As an exercise in learning distributed training, I pretrained a ~5.55M param
GPT-2 for ~2.3 epochs on
[BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus). I trained
over 4 GPUs in a single node using [Hugging Face
Accelerate](https://huggingface.co/docs/accelerate/en/index).

The model I trained is on the Hugging Face Hub
[here](https://huggingface.co/ishanjmukherjee/gpt2-micro/). Here is a
minimal script for doing inference with it:

```python
model_id = "ishanjmukherjee/gpt2-micro"

# Load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_id)

# Prompt
prompt = "To be or not to be,"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)

# Decode and print
print("---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
