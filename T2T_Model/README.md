## 📘 Documentation: Fine-tuning T5 on TruthfulQA (LLM Project)

---

### 🔹 **Cell 1: Clean Up and Install Dependencies**

```python
!rm -r ~/.cache/huggingface

!pip uninstall -y transformers tokenizers sentence-transformers
!pip cache purge

!pip install transformers==4.28.1 tokenizers==0.13.3 sentence-transformers==2.2.2

!pip install -U datasets evaluate
```

**Explanation:**

This cell resets the environment and installs the exact versions of the libraries I need:

- Removes Hugging Face’s cache to avoid version conflicts.
- Uninstalls older versions of `transformers`, `tokenizers`, and `sentence-transformers`.
- Reinstalls them with compatible versions for `Flan-T5`.
- Updates `datasets` and `evaluate` libraries for loading and assessing data.

---

### 🔹 **Cell 2: Disable W\&B Logging**

```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

**Explanation:**

I disable Weights & Biases (W\&B) logging to avoid unnecessary logging and tracking during training. It simplifies output and prevents warnings in Colab.

---

### 🔹 **Cell 3: Load Dataset**

```python
from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation", split="validation")

print(dataset[0])
```

**Explanation:**

- I’m loading the `"truthful_qa"` dataset from Hugging Face with the `"generation"` configuration.
- I use the `"validation"` split since there’s no `train` split in this dataset.
- Print a sample item to understand the structure. Each sample has:

  - `question`
  - `best_answer`
  - and other fields (e.g., `correct_answers`, `incorrect_answers`)

---

### 🔹 **Cell 4: Load Tokenizer**

```python
from transformers import AutoTokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Explanation:**

I’m using Google’s `flan-t5-base`, a fine-tuned variant of T5 with better reasoning and instruction-following ability.
The tokenizer converts questions and answers to token IDs for model input and supervision.

---

### 🔹 **Cell 5: Preprocessing Function**

```python
def preprocess(example):
    model_inputs = tokenizer(
        example["question"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["best_answer"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    labels["input_ids"] = [
        (token_id if token_id != tokenizer.pad_token_id else -100)
        for token_id in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

**Explanation:**

This function prepares the data for training:

- Tokenizes the `question` as input and `best_answer` as label.
- `max_length=128` ensures consistent size.
- Padding tokens in labels are replaced with `-100` so the loss function ignores them during training (important for T5).
- `as_target_tokenizer()` is necessary for T5 models since input and target use the same tokenizer but need separate handling.

---

### 🔹 **Cell 6: Tokenize the Dataset**

```python
tokenized_dataset = dataset.map(preprocess)
```

**Explanation:**

This line applies the `preprocess()` function to the entire dataset using `map()`.
The result is a dataset where both the input (`input_ids`) and output (`labels`) are tokenized and ready for training.

---

### 🔹 **Cell 7: Load Pretrained Model**

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

**Explanation:**

Loads the Flan-T5 model in a sequence-to-sequence (encoder-decoder) architecture, perfect for question-answer generation tasks.

---

### 🔹 **Cell 8: Set Training Arguments**

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qa-t5-model",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    evaluation_strategy="no"
)
```

**Explanation:**

Defines hyperparameters and training configuration:

- `output_dir`: Folder to save the model.
- `batch_size=8`: Good balance for Colab’s GPU.
- `num_train_epochs=10`: Trains for 10 full passes over the dataset.
- `logging_steps=10`: Logs every 10 steps.
- `save_steps=500`: Saves model every 500 steps.
- `save_total_limit=1`: Keeps only the latest checkpoint.
- `evaluation_strategy="no"`: No evaluation loop (I only fine-tune here).

---

### 🔹 **Cell 9: Train the Model**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

**Explanation:**

- Wraps the model, training arguments, and dataset into a `Trainer` object.
- Calls `.train()` to start the fine-tuning process using the tokenized dataset.

---

### 🔹 **Cell 10: Save the Model and Tokenizer**

```python
model.save_pretrained("./qa-t5-model")
tokenizer.save_pretrained("./qa-t5-model")
```

**Explanation:**

After training, I save both the fine-tuned model and tokenizer to the `qa-t5-model` directory so I can reload them later or export them.

---

### 🔹 **Cell 11: Create Inference Pipeline**

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

qa = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
model.eval()
```

**Explanation:**

- Sets up a Hugging Face `pipeline` for generation using my fine-tuned model.
- Automatically uses GPU (`device=0`) if available.
- `model.eval()` puts the model in evaluation mode (no dropout).

---

### 🔹 **Cell 12: Inference Test**

```python
output = qa("Why do veins appear blue?", max_length=64, num_beams=4, early_stopping=True)
print(output[0]["generated_text"])
```

**Explanation:**

Tests the model on a real-world prompt.

- `num_beams=4`: Enables beam search to improve output quality.
- `early_stopping=True`: Stops decoding once the most likely answer is found.
- `max_length=64`: Limits the generated response.

---

### 🔹 **Cell 13: Inspect 10 Raw Examples**

```python
for i in range(10):
  print(dataset[i])
```

**Explanation:**

Simple loop to print 10 examples from the dataset to manually verify that the model has seen relevant data during training.

---

### 🔹 **Cell 14: Zip the Model Folder**

```python
!zip -r qa-t5-model.zip qa-t5-model
```

**Explanation:**

Zips the model directory (`qa-t5-model`) into a `.zip` file for easier download.

---

### 🔹 **Cell 15: Download the Model**

```python
from google.colab import files
files.download("qa-t5-model.zip")
```

**Explanation:**

Uses Colab's `files.download()` to let me download the zipped model to my local machine.
