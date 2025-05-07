## 🏡 Forecasting House Prices Accurately Using Smart Regression Techniques in Data Science

> **Project Type**: Regression Problem
> **Dataset**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
> **Tech Stack**: Python, scikit-learn, LightGBM, XGBoost, Streamlit
> **Use Case**: Accurate house price prediction using statistical and ensemble ML techniques

---

### 🧠 Project Summary

This project presents a **smart regression model** to accurately forecast house prices based on various property characteristics. We leverage a blend of classical linear models (Ridge, Lasso) and powerful tree-based models (XGBoost, LightGBM), combined with extensive feature engineering and data cleaning techniques to produce highly reliable results.

The final model is deployed as a **Streamlit web app**, where users can input home features and get a real-time price prediction!

---

### ✨ Key Features

* 📊 Interactive **Exploratory Data Analysis** with visual insights
* 🛠️ Advanced **Data Cleaning & Feature Engineering**
* 🧪 Evaluated multiple regression models using **RMSE**
* 🏆 Final model selection based on cross-validation
* 🌐 **Streamlit App** for real-time prediction and GUI
* 💾 Trained models saved and reused for fast prediction

---

### 📂 Project Structure

```
HPP-GUI/
├── app.py                    # Streamlit application
└── data/
    ├── train.csv
    ├── test.csv
└── models/
    ├── ridge_model.pkl          # Final trained RidgeCV model
    ├── top_features.pkl         # List of top features used
    ├── all_features.pkl         # All one-hot encoded feature columns
├── result.csv           # Sample output predictions
├── README.md
├── requirements.txt         # Python dependencies
├── HousePricePrediction.ipynb  # Full analysis & model training
```

---

### 🔧 How It Works

1. **Preprocessing & Cleaning**

   * Fill missing values (median for numeric, "None" for categorical)
   * Feature engineering: `TotalSF`, `TotalBath`, log transformations
   * One-hot encoding for categoricals
2. **Modeling**

   * Linear models: RidgeCV, LassoCV
   * Tree models: LightGBM, XGBoost
   * Evaluation: RMSE on log-transformed prices
3. **Deployment**

   * Final model: RidgeCV
   * Streamlit GUI with top 15 features
   * Predicts and displays actual price using `np.expm1(log_price)`

---

### 🚀 How to Run Locally

#### 📥 1. Clone the Repo

```bash
git clone https://github.com/MTS-Krishna/HPP-GUI.git
cd HPP-GUI
```

#### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 🎯 3. Run the App

```bash
streamlit run app.py
```

> The app usually runs at: [http://localhost:8501](http://localhost:8501)

---

### 🛠 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
streamlit
joblib
```

(Already listed in `requirements.txt`)

---

### 📈 Model Performance

| Model    | RMSE (Log Scale)     |
| -------- | -------------------- |
| RidgeCV  | ✅ Best (lowest RMSE) |
| LassoCV  | Slightly worse       |
| XGBoost  | Competitive          |
| LightGBM | Competitive          |

---

### 🌐 Deployment Options

* ✅ Run locally with Streamlit
* ☁️ Deploy to [Streamlit Cloud](https://streamlit.io/cloud)
* 📦 Optionally dockerize for flexible deployment

---

### 🤝 Acknowledgements

* 📊 [Kaggle: House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* 💡 Inspiration & guidance from Data Science mentors and open-source contributors

---
