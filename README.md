# GPT-2 Micro

As an exercise in learning distributed training, I pretrained a ~5.55M param
GPT-2 on [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus),
over 4 GPUs in a single node using [Hugging Face
Accelerate](https://huggingface.co/docs/accelerate/en/index).
