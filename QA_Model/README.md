## 📘 Documentation: Fine-tuning DistilBERT on AdversarialQA Dataset

---

### 🔹 **Cell 1: Environment Reset and Install Dependencies**

```python
!rm -r ~/.cache/huggingface

!pip uninstall -y transformers tokenizers sentence-transformers
!pip cache purge

!pip install transformers==4.28.1 tokenizers==0.13.3 sentence-transformers==2.2.2

!pip install -U datasets evaluate
```

**Explanation:**

To ensure a clean and reproducible environment:

- I clear the Hugging Face cache.
- Uninstall potentially conflicting versions of key libraries.
- Install specific versions that are known to work well with `distilBERT` and Hugging Face's Trainer API.
- `datasets` and `evaluate` are updated to the latest versions to load and validate datasets.

---

### 🔹 **Cell 2: Disable Weights & Biases Logging**

```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

**Explanation:**

I turn off Weights & Biases logging to avoid tracking and reduce clutter in the output.

---

### 🔹 **Cell 3: Load Model and Tokenizer**

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

**Explanation:**

I use `distilbert-base-cased-distilled-squad`, a lightweight but powerful model trained for QA tasks.

- `AutoTokenizer` handles input preprocessing.
- `AutoModelForQuestionAnswering` is a model head specifically for span prediction (start & end positions of answers).

---

### 🔹 **Cell 4: Load AdversarialQA Dataset**

```python
dataset = load_dataset("adversarial_qa", "dbert")
```

**Explanation:**

Loads the `"adversarial_qa"` dataset using the `"dbert"` subset.
This dataset is specifically built to challenge QA models with more difficult examples.

---

### 🔹 **Cell 5: Preprocessing Function**

```python
def preprocess(example):
    ...
```

**Explanation:**

This function tokenizes inputs and computes start/end positions for the answer span.

Step-by-step:

- Tokenize the **question** and **context**.
- Keep the **offset mapping** to locate token positions corresponding to character spans.
- For each answer:

  - Convert the answer's character-level span into token-level span.
  - If no answer is provided, default both positions to 0.

- Finally, remove `offset_mapping` as it's no longer needed.

---

### 🔹 **Cell 6: Tokenize Entire Dataset**

```python
tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
```

**Explanation:**

I apply the `preprocess()` function to all samples in the dataset using `map()`:

- `batched=True` speeds up processing.
- Original columns are removed and replaced with tokenized tensors.

---

### 🔹 **Cell 7: Training Configuration**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    overwrite_output_dir=True,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)
```

**Explanation:**

These are the hyperparameters for training:

- Model is saved and evaluated at the end of each epoch.
- `learning_rate=2e-5` is typical for BERT-like models.
- `batch_size=8` is a safe size for Colab’s GPU.
- Logs are saved every 10 steps.

---

### 🔹 **Cell 8: Trainer Setup and Training**

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()
```

**Explanation:**

I use Hugging Face’s high-level `Trainer` API for training.
It handles batching, loss computation, optimization, evaluation, and saving checkpoints—all automatically.

---

### 🔹 **Cell 9: Inference Using Pipeline**

```python
from transformers import pipeline

qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

context = "Google Colab is a free platform that allows users to write and execute Python code in the browser."
question = "What is Google Colab?"

result = qa(question=question, context=context)
print("Answer:", result["answer"])
```

**Explanation:**

I test the fine-tuned model using the `question-answering` pipeline:

- Inputs are the question and its corresponding context.
- It returns the most likely answer span extracted from the context.
- `device=0` ensures it runs on GPU if available.

---

### 🔹 **Cell 10: Export Trained Model**

```python
from google.colab import files
import shutil

shutil.make_archive("trained_model", 'zip', "model")

files.download("trained_model.zip")
```

**Explanation:**

I zip the trained model directory (`model`) into `trained_model.zip` and trigger the download to save it locally.

⚠️ Note: You must make sure the model is saved into `model` before this step, or adjust the path accordingly.

