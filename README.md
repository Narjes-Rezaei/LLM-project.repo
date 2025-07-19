## 📄 Project Documentation – LLM-based Retrieval System

This project consists of two different models built and fine-tuned for natural language processing tasks using HuggingFace Transformers. Here's a complete explanation of the directory structure and model logic.

---

### 📂 Project Structure

```
.
├── QA_Model
│   ├── LLMProject.ipynb
│   ├── model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── README.md
│   ├── RunModel.ipynb
│   └── RUNMODEL.md
└── T2T_Model
    ├── LLMProject.ipynb
    └── README.md
```

#### 🔹 `QA_Model/`

This folder contains the **Question Answering model**.

- `LLMProject.ipynb`: The notebook where the QA model was trained and fine-tuned. It includes data loading, tokenizer setup, model loading, training loop, and evaluation.
- `RunModel.ipynb`: A separate notebook for testing the trained QA model. It takes user questions and a context as input, then predicts the answer.
- `model/`: Contains all the trained model artifacts, including:

  - `pytorch_model.bin`: The trained model weights.
  - `config.json`: Configuration of the model architecture.
  - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`: Tokenizer details.

- `README.md`: Basic info about the QA model project.
- `RUNMODEL.md`: Describes how to load and run inference on the trained QA model.

---

#### 🔹 `T2T_Model/`

This folder holds a simpler **Text-to-Text (T2T)** model.

- `LLMProject.ipynb`: In this notebook, the model is trained on a prompt-based dataset. It takes a question (prompt) and generates an answer without requiring any extra context.
- `README.md`: Describes how the T2T model works and how it was trained.

---

## 🔍 Difference Between QA and T2T Models

| Feature               | QA_Model                               | T2T_Model                       |
| --------------------- | -------------------------------------- | ------------------------------- |
| 🔄 Input              | Question + Context                     | Just a Question (Prompt)        |
| 🧠 Model Type         | Extractive QA (like BERT)              | Text generation model (like T5) |
| 🏷️ Use Case           | Needs context to answer properly       | Can generate free-text answers  |
| 📦 Output Type        | Span from context                      | Entirely new generated text     |
| 🔧 Training Objective | Find the start and end token of answer | Generate answer token-by-token  |

### ✅ When to use which?

- **QA_Model** is ideal when you have a clear **context document** and need to extract a specific answer from it. It performs best on structured QA datasets.
- **T2T_Model** is more flexible and can be used for **open-ended questions**, chatbots, summarization, or generation tasks. It doesn’t need an external context.
