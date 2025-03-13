# 📩📊 Spam Classification using 🤖 Naïve Bayes & 🔤 CountVectorizer

## 🌟 Overview
This 🏗️ project demonstrates a simple **📩 Spam Classification** system using **🤖 Naïve Bayes** & **🔤 CountVectorizer**. It processes 📝 text data, converts it into 🔢 numerical features, & classifies messages as **📩 Spam** or **✅ Ham (Not Spam)**.

## 📂 Dataset
The 📊 dataset contains labeled 📱 SMS messages, categorized as either **📩 Spam** or **✅ Ham**. You can use 📥 publicly available datasets like the **📚 SMS Spam Collection** from UCI 🏫.

### 📜 Dataset Columns
- **📌 Category:** Indicates if the 📝 message is 📩 spam or ✅ ham.
- **📝 Message:** The actual 🔤 text content.
- **⚠️ Spam:** A custom column 🏗️ by the user for classification.

## 🛠️ Features Used
- **🔤 CountVectorizer:** Converts 📜 text messages into a 👜 bag-of-words model.
- **🤖 Naïve Bayes Classifier:** A 🧮 probabilistic classifier based on 🏛️ Bayes' Theorem, great for 📩 text classification.

## 🏗️ Installation
To ⚙️ set up the project, install the required 📦 dependencies:

```bash
pip install numpy pandas scikit-learn
```

## 🏗️ Implementation
### 1️⃣ Import 📚 Libraries
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

### 2️⃣ Load 📂 Dataset
```python
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['Category', 'Message', 'Spam']]
data.columns = ['category', 'message', 'spam']
```

### 3️⃣ Preprocess 🏗️ Data
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ Train 🤖 Naïve Bayes Model
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 5️⃣ Evaluate 🏆 Model
```python
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
```

## 🛠️ Usage
Test the model with a custom 📩 message:
```python
message = ["🎉 Congratulations! You've won a 💰 lottery. Claim now."]
message_vectorized = vectorizer.transform(message)
prediction = model.predict(message_vectorized)
print("📩 Spam" if prediction[0] == 1 else "✅ Ham")
```

## 📊 Results
- 🎯 High accuracy in 📩 spam detection.
- 🚀 Can be improved with additional NLP preprocessing like ✂️ stopword removal & 🔄 stemming.

## 🔮 Future Improvements
- 🔍 Implement TF-IDF for feature extraction.
- 🧠 Use deep learning models like LSTMs for better accuracy.
- 🌍 Deploy as a web app using Flask or FastAPI.

## 📜 License
This 🏗️ project is open-source & available under the **📝 MIT License**.

---
Happy coding! 🚀😃

