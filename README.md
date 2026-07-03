# Employee Attrition Prediction System

An end-to-end data science and machine learning pipeline designed to predict employee attrition. By analyzing historical human resource metrics—such as job satisfaction, monthly income, age, work-life balance, and years at the company—this system helps organizations proactively identify flight risks and optimize retention strategies.

---

## 🚀 Features
* **Exploratory Data Analysis (EDA):** Complete data profiling via Jupyter Notebooks exploring correlations between employee demographics and turnover rates.
* **Predictive Modeling:** Trains a robust classification model capable of predicting the probability of an employee leaving the company.
* **Production Script:** Includes a standalone Python pipeline script for direct feature processing and model inference.
* **Presentation-Ready:** Comes with a comprehensive slide deck summarizing data insights, model evaluation metrics, and key performance indicators (KPIs).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science & ML:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook
* **Serialization:** Pickle (`.pkl`) for model persistence

---

## 📂 Project Structure
```text
├── Employee_Attrition_Prediction.ipynb       # Main exploratory data analysis & model training notebook
├── employee_attrition_classification.py       # Standalone Python script for deployment/inference
├── attrition_model.pkl                       # Serialized trained machine learning model
├── WA_Fn-UseC_-HR-Employee-Attrition.csv     # IBM HR analytics employee dataset
├── attrition.pptx                            # Project summary and insights presentation deck
├── .gitignore                                # Git ignore configurations
└── results/                                  # Exported visualization plots and model evaluation outputs
