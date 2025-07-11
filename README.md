# 🔍 Question Answering with Fine-Tuned RoBERTa on SQuAD v2

This project demonstrates how to fine-tune a pre-trained transformer model for **extractive question answering** using the [SQuAD v2 dataset](https://huggingface.co/datasets/squad_v2) and the model [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2), which is specifically designed for question answering tasks.

---

## 📦 Model & Dataset

- **Model:** `deepset/roberta-base-squad2`

  - A RoBERTa-based model trained on SQuAD2.0, capable of returning "no answer" if the context does not contain the answer.

- **Dataset:** `squad_v2` from Hugging Face Datasets

  - Consists of context-question pairs, some of which have no answer in the context (unanswerable questions).

---

## 🛠️ Steps Performed

### 1. **Setup Environment**

Installed the required libraries in Google Colab:

```python
!pip install transformers datasets evaluate
```

### 2. **Load Pre-trained Model and Dataset**

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

dataset = load_dataset("squad_v2")
```

---

### 3. **Preprocess the Dataset**

Tokenized the dataset with proper padding, truncation, and length handling:

```python
def preprocess(example):
    return tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )

tokenized_datasets = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

---

### 4. **Configure Training Parameters**

Used `TrainingArguments` from Hugging Face to define training settings:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)
```

---

### 5. **Train the Model**

Used the Hugging Face `Trainer` API for training:

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

---

### 6. **Save the Trained Model**

After training, saved the fine-tuned model:

```python
model.save_pretrained("project_code/model")
tokenizer.save_pretrained("project_code/model")
```

---

### 7. **Test the Trained Model**

Used the pipeline API to ask questions from a context:

```python
from transformers import pipeline

qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = "Google Colab is a free Jupyter notebook environment that runs in the cloud."
question = "What is Google Colab?"

result = qa(question=question, context=context)
print(result["answer"])
```
