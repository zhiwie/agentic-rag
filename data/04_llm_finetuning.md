# LLM Fine-Tuning

## What is fine-tuning?

Fine-tuning is the process of taking a pre-trained large language model and training it
further on a smaller, specialized dataset. The goal is to adapt the model to a specific
domain, task, or style. For example, a general model can be fine-tuned on medical text
to make it better at answering medical questions, or on a company's support tickets to
make it follow that company's tone and procedures.

## Fine-tuning versus RAG

Fine-tuning and RAG solve different problems and are often confused. Fine-tuning changes
the model's behavior and is best for teaching the model a skill, format, or style. RAG
gives the model access to facts at query time and is best for knowledge that changes
frequently or is too large to memorize. A common guideline is: use RAG when you need the
model to know new facts, and use fine-tuning when you need the model to behave in a new
way. Many production systems combine both.

## Full fine-tuning versus parameter-efficient methods

Full fine-tuning updates every weight in the model. It is powerful but expensive,
because it requires storing and updating billions of parameters and large amounts of GPU
memory. Parameter-efficient fine-tuning (PEFT) methods update only a small subset of
parameters. The most popular PEFT method is LoRA (Low-Rank Adaptation), which freezes
the original weights and inserts small trainable matrices. LoRA dramatically reduces
memory requirements and makes it possible to fine-tune large models on more modest
hardware.

## Memory as a bottleneck

The main practical barrier to fine-tuning large models is GPU memory. The model weights,
the gradients, and the optimizer state must all fit in memory during training. This is
why fine-tuning very large models has traditionally required expensive multi-GPU
clusters. Techniques that offload part of this memory burden to other storage, or that
reduce the memory footprint through quantization and low-rank methods, make on-premise
fine-tuning more accessible to smaller organizations.

## On-premise fine-tuning

Running fine-tuning on-premise, rather than in the cloud, keeps proprietary data inside
an organization's own infrastructure. This is important for industries with strict data
privacy, security, or sovereignty requirements. Solutions that pair efficient
fine-tuning techniques with high-throughput local storage aim to make on-premise LLM
training practical without requiring a massive GPU investment.
