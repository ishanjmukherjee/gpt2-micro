import os
from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint

def main():
    # load tokenized dataset
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # causal LM, not masked LM
    )
    ds = load_from_disk("/scratch/hsulab/ishanjmukherjee/gpt2-micro-dataset")

    # model config ~5.5 M params
    config = GPT2Config(
        n_positions = 512,
        n_embd = 96,
        n_layer = 6,
        n_head = 6, # must divide n_embd (96/6 = 16 dims per head)
    )
    model = GPT2LMHeadModel(config)

    # training args
    args = TrainingArguments(
        output_dir="outputs",
        overwrite_output_dir=True,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=8,
        bf16=True,
        max_steps=1200,
        logging_steps=1,
        save_steps=5,
        report_to="wandb",
        run_name="gpt2-micro-pretrain",
        push_to_hub=True,
        hub_model_id="ishanjmukherjee/gpt2-micro",
        hub_strategy="every_save",
        hub_private_repo=False
    )

    last_cp = None
    if os.path.isdir(args.output_dir):
        last_cp = get_last_checkpoint(args.output_dir)
        if last_cp:
            print(f">>> Resuming from checkpoint: {last_cp}")
        else:
            print(">>> No existing checkpoint found. Starting fresh.")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=last_cp)

if __name__=="__main__":
    main()
