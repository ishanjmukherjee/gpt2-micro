from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast

def main():
    # load
    books = load_dataset("bookcorpus", split="train", trust_remote_code=True)

    # shuffle
    books = books.shuffle(seed=42)

    # tokenize
    print("Tokenizing...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = books.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=False,
            return_attention_mask=False
        ),
        batched=True,
        remove_columns=["text"],
        num_proc=16
    )

    # pack
    print("Packing...")
    def group_texts(examples, block_size=512):
        # Concatenate then split into fixed-size chunks
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        chunks = [
            concatenated[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]
        return {"input_ids": chunks}
    chunked = tokenized.map(
        group_texts, batched=True, num_proc=16
    )

    chunked.save_to_disk("/scratch/hsulab/ishanjmukherjee/gpt2-micro-dataset")

if __name__=="__main__":
    main()
