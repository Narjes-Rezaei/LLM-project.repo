## 💻 1. Installing Jupyter Notebook on Ubuntu

First, if `Python` and `pip` are not already installed on your system, install them with:

```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

Now install Jupyter Notebook:

```bash
pip3 install notebook
```

To run Jupyter:

```bash
jupyter notebook
```

🔗 Once launched, it will open in your browser at:

```
http://localhost:8888
```

---

## 📦 2. Installing Required Libraries for the Model

In a Jupyter cell or terminal, install the necessary Python packages:

```python
!pip3 install transformers datasets torch
```

---

## 🤖 3. Running the Trained Model in Jupyter Notebook

Let’s assume your trained model is saved in a folder called `model` and it’s located in the same directory as your notebook.

Here’s an explanation of the file `RunModel.ipynb`:

---

```python
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

model_path = "model"

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
```

🧾 **Explanation:**

- First, we import the required modules.
- `model_path` defines where the saved model is stored.
- We load both the model and its tokenizer from that path.
- Then, we initialize a `question-answering` pipeline using Hugging Face Transformers.
- `device=-1` means it runs on the CPU (change to `0` for GPU).

---

```python
context = "A generation later, the Irish Anglican bishop, George Berkeley (1685–1753), determined that Locke's view immediately opened a door that would lead to eventual atheism. In response to Locke, he put forth in his Treatise Concerning the Principles of Human Knowledge (1710) an important challenge to empiricism in which things only exist either as a result of their being perceived, or by virtue of the fact that they are an entity doing the perceiving. (For Berkeley, God fills in for humans by doing the perceiving whenever humans are not around to do it.) In his text Alciphron, Berkeley maintained that any order humans may see in nature is the language or handwriting of God. Berkeley's approach to empiricism would later come to be called subjective idealism."
question = "what group is mentioned last?"
```

🧾 **Explanation:**

- `context` is the passage of text from which the model should extract the answer.
- `question` is the user’s query related to that context.
- These are the two primary inputs passed to the pipeline.

---

```python
result = qa(question=question, context=context)
print("Answer:", result["answer"])
```

🧾 **Explanation:**

- The `qa(...)` call runs inference using the question-answering model.
- The output (most likely answer) is stored in `result`.
- The answer is then printed.

---

## 📚 4. Where to Get More Questions and Contexts?

If you'd like to try real examples from the same dataset your model was trained on (like `adversarial_qa`), you can view them directly on Hugging Face here:

🔗 [https://huggingface.co/datasets/UCLNLP/adversarial_qa/viewer/dbert](https://huggingface.co/datasets/UCLNLP/adversarial_qa/viewer/dbert)

