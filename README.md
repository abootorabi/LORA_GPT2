# LoRA Fine-Tuning for Question Answering

This repository contains an implementation of a **parameter-efficient fine-tuning** pipeline for a large language model (LLM), specifically adapted for question answering on the SQuAD dataset. The project leverages **Low-Rank Adaptation (LoRA)** to significantly reduce the number of trainable parameters while maintaining high performance. 🧠

---

## Project Overview

Our approach involves fine-tuning a **GPT-2 model** on the **SQuAD dataset** using **LoRA**. This method is explored to understand the theoretical foundations and practical implications of modern fine-tuning techniques. The complete pipeline includes **data preprocessing**, **custom data collation**, **model training with LoRA**, and **comprehensive evaluation**.

---

## Key Features

* **Parameter-Efficient Fine-Tuning**: Implementation and analysis of **LoRA** to reduce computational costs while preserving model performance. LoRA constrains weight updates to a low-rank decomposition, where only the matrices **A** and **B** are trained while the original weight matrix **W** remains frozen.
* **Dataset Preprocessing**: Advanced tokenization strategies for sequence-to-sequence tasks, including proper attention masking. Examples are formatted with structured prompts containing the **context**, **question**, and **answer**, with the loss function masking prompt tokens to focus on answer generation.
* **Custom Training Pipeline**: An end-to-end model training process with a custom data collator for efficient batching, dynamic padding, and label masking.
* **Comprehensive Model Evaluation**: The model's performance is assessed using domain-specific metrics like **SQuAD F1** and **Exact Match**.
* **Inference and Generation**: The project includes the implementation of controlled text generation with various sampling strategies, such as greedy decoding, Top-k sampling, and Nucleus sampling, to analyze the impact on answer quality and diversity.

---

## Theoretical Background

### Low-Rank Adaptation (LoRA)

**LoRA** reduces the number of trainable parameters by representing the weight update as a low-rank decomposition. The formula for this is $W' = W + \Delta W = W + BA$, where **W** is the original pre-trained weight matrix, and $\Delta W = BA$ is the low-rank decomposition. The matrices **A** and **B** are trained, while the original weights **W** are frozen.

### Question Answering with Causal Language Models

For extractive question answering, examples are formatted as:

`Context: [context_text] Question: [question_text] Answer: [answer_text]<eos>`

The loss function is designed to mask the prompt tokens, ensuring that learning is focused solely on generating the answer. This is represented by the formula:

$L = -\sum_{t=T_{prompt}}^{T_{total}} \log P(x_t|x_{<t})$
