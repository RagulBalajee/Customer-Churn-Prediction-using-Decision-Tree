# 📊 Customer Churn Prediction using Decision Tree

A Machine Learning project that predicts whether a telecom customer is likely to **stay with the company or leave (churn)** using a **Decision Tree Classifier**.

The project demonstrates the complete basic Machine Learning workflow — data generation, data exploration, visualization, train-test splitting, model training, prediction, evaluation, and Decision Tree visualization.

---

## 🎯 Problem Statement

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave can help companies take preventive actions such as improving customer support, offering suitable plans, or providing retention offers.

This project builds a classification model to predict customer churn.

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Customer stays  |
| `1`   | Customer leaves |

---

## 🚀 Project Overview

The project uses a synthetic dataset containing **200 customers** and multiple customer-related features.

The dataset was deliberately designed to contain:

* Multiple features
* Different feature scales
* Some random noise
* Overlapping classes
* Features that are not perfectly predictive

The churn target is generated using multiple factors such as support calls, payment delays, tenure, satisfaction score, and contract length, along with random noise.

---

## 📁 Dataset Features

The dataset contains **200 rows and 11 columns** — 10 input features and 1 target variable.

| Feature              | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `age`                | Customer age                                            |
| `tenure_months`      | Number of months the customer has been with the company |
| `monthly_charges`    | Customer's monthly charges                              |
| `total_charges`      | Customer's total charges                                |
| `support_calls`      | Number of customer support calls                        |
| `data_usage_gb`      | Data usage in GB                                        |
| `contract_length`    | Contract duration in months                             |
| `payment_delay`      | Number of payment-delay days                            |
| `satisfaction_score` | Customer satisfaction score                             |
| `referrals`          | Number of customer referrals                            |
| `churn`              | Target variable: 0 = Stay, 1 = Churn                    |

The generated dataset contains **123 customers who stay and 77 customers who churn**.

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 🤖 Scikit-learn
* 📓 Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Generate Dataset
       ↓
Understand the Data
       ↓
Check Data Types
       ↓
Check Missing Values
       ↓
Explore Class Distribution
       ↓
Visualize Data
       ↓
Separate Features & Target
       ↓
Train-Test Split
       ↓
Train Decision Tree
       ↓
Make Predictions
       ↓
Evaluate Model
       ↓
Visualize Decision Tree
```

---

## 🧠 Machine Learning Model

The project uses:

### Decision Tree Classifier

A Decision Tree is a supervised Machine Learning algorithm used for classification and regression tasks.

The initial model uses the **Gini impurity criterion** with `random_state=42`.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
```

The model is then trained using:

```python
model.fit(X_train, Y_train)
```

---

## ✂️ Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to preserve the class distribution.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
```

This results in **160 training samples and 40 testing samples**.

---

## 📈 Model Evaluation

The trained Decision Tree was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Classification Report

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |  **72.5%** |
| Precision | **66.67%** |
| Recall    | **53.33%** |
| F1 Score  | **59.26%** |

These are the results obtained from the notebook's final model evaluation.

### Confusion Matrix

```text
[[21  4]
 [ 7  8]]
```

The classification report shows:

```text
              precision    recall  f1-score   support

           0       0.75      0.84      0.79        25
           1       0.67      0.53      0.59        15

    accuracy                           0.72        40
   macro avg       0.71      0.69      0.69        40
weighted avg       0.72      0.72      0.72        40
```

---

## 🌳 Decision Tree Optimization

The notebook also experiments with limiting the depth of the Decision Tree.

The final model shown in the notebook uses:

```python
DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)
```

The purpose of limiting the tree depth is to control model complexity and reduce the possibility of excessive tree growth.

---

## 📊 Data Visualization

The project includes visualization using Matplotlib to explore relationships between features and churn.

For example, the notebook visualizes the relationship between:

```text
Support Calls → Churn
```

using a scatter plot.

The Decision Tree is also visualized to understand the model's decision-making structure.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## ▶️ How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Customer Churn Prediction using Decision Tree(2).ipynb
```

Then run the notebook cells from top to bottom.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── Customer Churn Prediction using Decision Tree(2).ipynb
│
└── README.md
```

---

## 🔍 Key Learning Outcomes

Through this project, you can understand:

* How to create and work with a dataset using Pandas
* How NumPy can be used for data generation
* How to inspect a dataset using `shape()`, `info()`, and missing-value checks
* How to visualize relationships in data
* How to separate features and target variables
* How to split data into training and testing sets
* How to build a Decision Tree Classifier
* How to make predictions using a trained model
* How to evaluate a classification model
* How to interpret a confusion matrix
* How precision, recall, and F1 score are calculated
* How Decision Tree depth affects model complexity

---

## ⚠️ Limitations

This project uses a **synthetically generated dataset** rather than a real-world telecom customer dataset.

Therefore, the reported performance should not be interpreted as the expected performance of a real telecom churn prediction system.

The dataset was generated using predefined relationships between several features and churn, with additional random noise.

---

## 🚀 Future Improvements

Possible improvements include:

* Use a real-world telecom churn dataset
* Perform feature engineering
* Handle categorical variables
* Compare multiple Machine Learning algorithms
* Tune Decision Tree hyperparameters
* Use GridSearchCV or RandomizedSearchCV
* Apply cross-validation
* Handle class imbalance
* Add feature importance analysis
* Compare Decision Tree with Random Forest
* Compare Logistic Regression, KNN, SVM, and Gradient Boosting
* Build a simple web application for real-time churn prediction
* Deploy the trained model using Flask or FastAPI

---

## 📌 Conclusion

This project demonstrates how a **Decision Tree Classifier** can be used to predict customer churn based on customer characteristics such as tenure, charges, support calls, payment delays, satisfaction, contract length, and referrals.

The final model achieved an accuracy of **72.5%** on the 40-sample test set, with a precision of **66.67%**, recall of **53.33%**, and F1 score of **59.26%**.

The project provides a practical introduction to **classification, model evaluation, and Decision Tree-based Machine Learning**.

---

## 👨‍💻 Author

**Ragul Balajee G.K**

Computer Science Engineering Student

---

⭐ If you found this project useful, consider giving the repository a star!
