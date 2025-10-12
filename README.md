# Pretraining-a-transformer-model-from-scratch-on-Wikidata
Unsupervised pretraining a Transformer GPT-2 style causal language model from scratch using Wikidata

---

## Pretraining a Transformer Model from Scratch on Wikidata

### Overview

This project walks through the **end-to-end process of pretraining a Transformer language model from scratch** using Wikidata.
This project looks into how LLMs acquire language understanding from text-only training.

---

### What This Project Does

1. **Data Acquisition via Kaggle API**
   The workflow starts by connecting to the Kaggle API, authenticating with a Kaggle key, and downloading JSONL datasets from Wikidata.

2. **Dataset Preparation & Streaming**
   Using Hugging Face’s `datasets` library, multiple JSONL dumps are streamed, combined, and cleaned.
   Each record’s `description`, `abstract`, and `sections` are concatenated to create training samples, filtering out short entries.
   >Text preprocessing, chunking to avoid running out of memory, large-scale streaming.

3. **Model Initialization & Configuration**
   A **Transformer architecture** is initialized using Hugging Face’s `transformers` library, specifying hyperparameters such as embedding dimension, hidden size, attention    heads, and layers.
   >Transformer (self-attention, positional encoding).

4. **Pretraining with Masked Language Modeling**
   The model is pretrained using a **Causal Language Modeling**.
   Training leverages the `Trainer` API with **AdamW optimization**, **cosine learning rate scheduler**, and **gradient accumulation**.

5. **Visualization**
   Training metrics are logged with **TensorBoard**, providing visual insights into learning dynamics and convergence.
---

### Model Architecture Diagram

Kaggle Dataset (Wikidata JSONL)
        >
Hugging Face Datasets (streaming)
        >
Text cleaning & merging (description + abstract + sections)
        >
Transformer model init (emb dim, layers, heads)
        >
Pretraining (CLM)
        >
Trainer: AdamW, LR scheduler, grad accumulation
        >
TensorBoard logging & checkpoints

---

### To use the model, you can

1. Clone the repository
2. Install dependencies
3. Set up Kaggle API
4. Download and prepare dataset
5. Run the notebook


### Results & Insights

* Understands French syntax pretty well
* Produces contextually relevant completions (“film sorti en 1988…”, “philosophie politique…”, “intelligence artificielle…”),
* But lacks coherence and tends to loop/repeat, which is common in small language models.
     Possible issue: dataset and models size are two small; no supervised fine-tuning

---

### Future Improvements

* Extend pretraining to larger multilingual corpora
* Fine-tune on downstream tasks (QA, summarization)



