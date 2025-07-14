# آموزش نصب و اجرای Jupyter Notebook روی اوبونتو و اجرای مدل یادگیری ماشین

## ۱. نصب Jupyter Notebook روی اوبونتو

1. **برو به ترمینال** (Ctrl+Alt+T)

2. نصب پایتون و pip (اگر نصب نیست):

```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

3. نصب Jupyter Notebook با pip:

```bash
pip3 install notebook
```

4. اجرای Jupyter Notebook:

```bash
jupyter notebook
```

* پس از اجرای دستور بالا، یک آدرس URL به شما داده می‌شود (مثل `http://localhost:8888/?token=...`).
* این آدرس را در مرورگر باز کنید تا وارد محیط Jupyter شوید.

---

## ۲. آماده‌سازی و بارگذاری مدل یادگیری ماشین در Jupyter

فرض می‌کنیم مدل شما فایل‌هایی دارد مانند `pytorch_model.bin`، `config.json` و ... در مسیر مشخص (مثلاً `project_code/model/`).

### ۱. نصب کتابخانه‌های مورد نیاز

مثلاً برای PyTorch و Transformers:

```bash
pip3 install torch transformers
```

### ۲. بارگذاری مدل در یک نوت‌بوک جدید

در Jupyter، یک نوت‌بوک Python جدید بسازید و کد زیر را برای بارگذاری مدل اجرا کنید:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "project_code/model"

# بارگذاری توکنایزر
tokenizer = AutoTokenizer.from_pretrained(model_path)

# بارگذاری مدل
model = AutoModelForCausalLM.from_pretrained(model_path)

# تست مدل با یک ورودی نمونه
input_text = "سلام، حال شما چطور است؟"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```

---

## ۳. کانتکست و سوال چیست و از کجا می‌آید؟

* **سوال (Query):** سوالی است که کاربر به مدل می‌دهد تا مدل به آن پاسخ دهد.
* **کانتکست (Context):** مجموعه‌ای از اطلاعات یا داده‌ها است که مدل برای فهم بهتر سوال و تولید پاسخ دقیق‌تر از آن استفاده می‌کند. مثلاً متن یک مقاله، پاراگراف‌های مرتبط یا اطلاعات پیش‌زمینه.

---

### چگونه کانتکست و سوال را به مدل بدهیم؟

1. **سوال** معمولاً به صورت رشته (string) مستقیم به مدل داده می‌شود.

2. **کانتکست** می‌تواند به چند شکل باشد:

   * در همان رشته سوال اضافه شود. مثلاً:

   ```python
   input_text = "متن کانتکست ... \n سوال: ..."
   ```

   * یا به صورت جداگانه به مدل ارسال شود (اگر مدل پشتیبانی کند).

3. **منبع کانتکست:** می‌تواند از فایل‌های متنی، دیتاست، پایگاه داده، یا حتی پاسخ مدل‌های دیگر استخراج شود.