```python
!rm -r ~/.cache/huggingface
```

* این دستور حافظه کش محلی Hugging Face را پاک می‌کند.
* در مواقعی استفاده می‌شود که فایل‌های خراب یا نسخه‌های ناهماهنگ باعث خطا می‌شوند.
* باعث می‌شود مدل‌ها و توکنایزرها دوباره از اول دانلود شوند.

---

```python
!git clone https://github.com/Narjes-Rezaei/LLM-project.repo.git
```

* این خط، یک مخزن گیت (GitHub repository) را کلون می‌کند.
* تمام فایل‌های موجود در آن مخزن را به محیط فعلی (مثلاً Google Colab) کپی می‌کند.
* آدرس داده‌شده به عنوان منبع پروژه استفاده می‌شود.

---

```python
!pip uninstall -y transformers tokenizers sentence-transformers
!pip cache purge
```

* این دستور پکیج‌های `transformers`، `tokenizers`، و `sentence-transformers` را حذف می‌کند.
* این کار زمانی مفید است که بخواهیم نسخه‌های مشخص و بدون تداخل از این کتابخانه‌ها را نصب کنیم.
* همچنین `pip cache purge` کش pip را پاک می‌کند تا نسخه‌های قبلی در نصب جدید دخالت نکنند.

---

```python
!pip install transformers==4.28.1 tokenizers==0.13.3 sentence-transformers==2.2.2
```

* با این دستور نسخه خاصی از کتابخانه‌های HuggingFace نصب می‌شود.
* `transformers==4.28.1` نسخه پایدار مورد نیاز پروژه است.
* `sentence-transformers` و `datasets` برای کار با داده‌ها و بردارهای معنایی به کار می‌روند.

---

```python
!pip install -U datasets evaluate
```

* این دستور کتابخانه‌های `datasets` و `evaluate` را نصب یا به‌روزرسانی می‌کند.
* برای بارگذاری دیتاست‌ها و ارزیابی عملکرد مدل استفاده می‌شود.

---

```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

* تنظیم یک متغیر محیطی برای غیرفعال کردن اتصال به Weights & Biases (wandb).
* این کار از باز شدن پنجره لاگ‌گیری wandb جلوگیری می‌کند.

---

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

dataset = load_dataset("adversarial_qa", "dbert")
```

* بارگذاری مدل پایه `distilbert` مخصوص پرسش و پاسخ.
* دانلود توکنایزر و مدل آموزش‌دیده از Hugging Face.
* لود دیتاست `adversarial_qa` با کانفیگ `dbert` که شامل سؤالات چالشی است.

---

```python
def preprocess(example):
    questions = [q.strip() for q in example["question"]]
    contexts = example["context"]
    answers = example["answers"]

    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        start_positions.append(start_char)
        end_positions.append(end_char)

    tokenized_example = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    tokenized_example["start_positions"] = start_positions
    tokenized_example["end_positions"] = end_positions

    return tokenized_example
```

* تابع پیش‌پردازش داده‌ها برای مدل.
* سؤالات و متون را توکنایز می‌کند.
* موقعیت دقیق شروع و پایان پاسخ‌ها در توکن‌ها محاسبه می‌شود.
* آماده‌سازی داده برای آموزش مدل.

---

```python
tokenized_datasets = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

* اعمال تابع `preprocess` روی کل دیتاست.
* ستون‌های اصلی مثل `question`, `context` و `answers` حذف می‌شوند.
* نتیجه: دیتاست فقط شامل `input_ids`, `attention_mask`, `start_positions`, و `end_positions`.

---

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch"
)
```

* تنظیمات مربوط به آموزش مدل:

  * مسیر ذخیره نتایج، تعداد epoch، batch size، نرخ یادگیری و ...
  * مدل بعد از هر epoch ذخیره می‌شود.

---

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)
```

* آماده‌سازی Trainer برای مدیریت آموزش.
* شامل مدل، داده‌ی آموزش و ارزیابی، و تنظیمات از قبل تعیین‌شده.

---

```python
trainer.train()
```

* آغاز فرایند آموزش مدل.
* خروجی شامل لاگ‌های مربوط به loss و پیشرفت آموزش است.

---

```python
model.save_pretrained("project_code/model")
tokenizer.save_pretrained("project_code/model")
```

* ذخیره مدل و توکنایزر آموزش‌دیده در پوشه `project_code/model`.
* این فایل‌ها بعداً در حالت inference استفاده خواهند شد.

---

```python
from transformers import pipeline

qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = "Google Colab is a free platform that allows users to write and execute Python code in the browser."
question = "What is Google Colab?"

result = qa(question=question, context=context)
print("Answer:", result["answer"])
```

* استفاده از مدل آموزش‌دیده برای پاسخ به سؤال دلخواه.
* با `pipeline` راحت می‌توان مدل را تست کرد.
* خروجی نهایی چاپ می‌شود.