## 💻 ۱. نصب Jupyter Notebook روی Ubuntu

ابتدا اگر `Python` و `pip` نصب نیست، نصبش کن:

```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

حالا Jupyter را نصب کن:

```bash
pip3 install notebook
```

اجرا:

```bash
jupyter notebook
```

🔗 پس از اجرا، مرورگر باز می‌شود با آدرس:

```
http://localhost:8888
```

---

## 📦 ۲. نصب کتابخانه‌های مورد نیاز مدل

داخل یک سلول یا ترمینال، این پکیج‌ها را نصب کن:

```python
!pip3 install transformers datasets torch
```

---

## 🤖 ۳. اجرای مدل آموزش‌دیده در Jupyter Notebook

فرض می‌گیریم پوشه `project_code/model` که شامل مدل آموزش‌دیده هست، کنار فایل نوت‌بوک قرار دارد.

در اینجا، فایل `RunModel.ipynb` را توضیح می‌دهم:

---

```python
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

model_path = "project_code/model"

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
```

🧾 **توضیح:**

* ابتدا ماژول‌های مورد نیاز ایمپورت می‌شوند.
* `model_path` مسیر مدل ذخیره‌شده روی دیسک است.
* مدل و توکنایزر از همین مسیر لود می‌شوند.
* با `pipeline` نوع `question-answering` ساخته می‌شود.
* `device=-1` یعنی مدل روی CPU اجرا می‌شود (نه GPU).

---

```python
context = "A generation later, the Irish Anglican bishop, George Berkeley (1685–1753), determined that Locke's view immediately opened a door that would lead to eventual atheism. In response to Locke, he put forth in his Treatise Concerning the Principles of Human Knowledge (1710) an important challenge to empiricism in which things only exist either as a result of their being perceived, or by virtue of the fact that they are an entity doing the perceiving. (For Berkeley, God fills in for humans by doing the perceiving whenever humans are not around to do it.) In his text Alciphron, Berkeley maintained that any order humans may see in nature is the language or handwriting of God. Berkeley's approach to empiricism would later come to be called subjective idealism."
question = "what group is mentioned last?"
```

🧾 **توضیح:**

* `context`: متنی است که پاسخ سؤال باید از دل آن بیرون کشیده شود.
* `question`: سؤال مربوط به متن است.
* در پروژه ما، این‌ها همان ورودی کاربر هستند که pipeline با آن کار می‌کند.

---

```python
result = qa(question=question, context=context)
print("Answer:", result["answer"])
```

🧾 **توضیح:**

* `qa(...)` مدل را فراخوانی می‌کند و پاسخ را پیدا می‌کند.
* نتیجه در متغیر `result` ذخیره می‌شود و چاپ می‌شود.

---

## 📚 ۴. کوئری و context از کجا بیاریم؟

اگر بخوای کوئری‌هایی مشابه دیتاستی که آموزش دیدی (مثل `adversarial_qa`) استفاده کنی:

می‌تونی دیتاست رو از HuggingFace ببینی:
   🔗 [Adversarial QA Dataset](https://huggingface.co/datasets/adversarial_qa)