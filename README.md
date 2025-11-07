
# 📌 Machine Learning & NLP Classification Project

## 📖 Title & Short Description
This project focuses on applying **Natural Language Processing (NLP)** and **Supervised Machine Learning** techniques to classify text data into different categories. The primary goal is to evaluate and compare the performance of two widely used classification models:

- **Logistic Regression**
- **Random Forest Classifier**

These models are trained on TF‑IDF vectorized text features. This task is important because text classification plays a crucial role in various modern applications like:

✅ Sentiment analysis  
✅ Spam detection  
✅ Topic classification  
✅ Customer feedback analytics  

---

## 📊 Dataset Source & Preprocessing
The dataset used in this project is a **user‑provided CSV file** that contains:  
- A **text column**: natural language input
- A **categorical target label**: class for each text instance

### ✅ Preprocessing Applied:
- Removal of missing values
- TF‑IDF vectorization of text for numerical representation
- Label Encoding for target column
- Train‑Test split (80% training / 20% testing)

No heavy filtering or cleaning was required, making the dataset suitable for baseline NLP experiments.

---

## 🧠 Methods & Approach

### 🔹 Technique Used
| Stage | Method |
|-------|-------|
| Text → Numeric conversion | **TF‑IDF Vectorizer** |
| Learning algorithms | **Logistic Regression** & **Random Forest** |
| Evaluation | Accuracy + Macro F1‑Score |

### ✅ Why This Approach?
- **TF‑IDF** captures word importance and reduces noise
- **Logistic Regression** is a strong baseline for linear text classification tasks
- **Random Forest** captures non‑linear relationships and provides robust performance

We compared two fundamentally different modeling strategies to understand how linear vs. tree‑based classifiers behave on the same text data.

---

## ▶️ Steps to Run the Code

### 🔧 Requirements Installation
```bash
pip install pandas scikit-learn
```

### ▶️ Run the Script
```bash
python your_script.py
```

OR inside Jupyter/Colab simply execute all cells.

---

## 🧪 Experiments & Results Summary

The models were evaluated using:

- ✅ Accuracy  
- ✅ Macro F1‑Score (balanced metric for multiple classes)

| Model | Accuracy | F1‑Score (Macro) |
|--------|:-------:|:---------------:|
| Logistic Regression | High performance on linear separable text | Performs very well |
| Random Forest | Competitive accuracy | Slightly lower than LR for sparse vectors |

> 📌 Visualization techniques such as bar charts and tables were used to gain insight into model performance.

✅ Logistic Regression performed slightly better overall due to the sparse nature of TF‑IDF vectors.  
✅ Random Forest still remained a strong alternative with robust generalization ability.

---

## ✅ Conclusion

From this project, we learned that:
- TF‑IDF is an effective technique for transforming textual data into numerical features
- Logistic Regression tends to perform best on sparse, high‑dimensional NLP datasets
- Random Forest still holds strong performance without complex hyperparameter tuning

📌 Future improvements may include:
- Advanced text cleaning (lemmatization, stopword removal)
- Hyperparameter tuning (GridSearch/RandomSearch)
- Deep Learning models such as BERT or LSTMs
- Visualization of feature importances and confusion matrices

---

## 📚 References
1️⃣ Scikit‑Learn Documentation: https://scikit-learn.org  
2️⃣ TF‑IDF Article: https://en.wikipedia.org/wiki/Tf–idf  
3️⃣ Logistic Regression in ML: https://developers.google.com/machine-learning  
4️⃣ Random Forest: Breiman, L. (2001). Machine Learning Journal  

---

✅ Project completed using **Python, NLP & Machine Learning Models**  
