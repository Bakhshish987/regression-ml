# 🏠 House Price Prediction with Lasso Regression

Predicting residential housing prices in Ames, Iowa using the Kaggle "House Prices: Advanced Regression Techniques" dataset.  
This project applies regularized linear models and feature engineering to develop a robust regression pipeline.

---

## 📊 Project Overview

- Performed data exploration and handled missing values
- Applied log transformation to normalize the target variable (`SalePrice`)
- Used one-hot encoding for categorical features
- Trained Lasso and Ridge regression models using scikit-learn
- Evaluated models using RMSE via 5-fold cross-validation
- Achieved a **0.13450 RMSLE** on the Kaggle public leaderboard 🎯

---

## 🧠 Techniques Used

- `pandas`, `numpy` for data wrangling
- `matplotlib`, `seaborn` for EDA
- `get_dummies`, `.fillna()` for preprocessing
- `Lasso`, `Ridge`, `LinearRegression` from `sklearn`
- `cross_val_score`, `mean_squared_error` for evaluation
- Log transformation: `np.log1p()` and inverse: `np.expm1()`

---

## 📁 Dataset

- [Kaggle House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- `train.csv`, `test.csv`, `submission.csv`

---

## 🚀 How to Run

1. Clone this repo or download the notebook
2. Run all cells to reproduce results
3. Submit `submission.csv` to Kaggle to compare scores

---

## 🏁 Results

| Model         | RMSE (CV) |
|---------------|-----------|
| Linear        | 0.1551    |
| Ridge (α=10)  | 0.1393    |
| Lasso (α=0.001) | **0.1385**  |

🧾 **Leaderboard Score**: 0.13450 (Lasso)

---

## 🔮 Future Work

- Hyperparameter tuning via `GridSearchCV`
- Feature engineering (e.g., extract year-related trends)
- Try tree-based models like `XGBoost` and `LightGBM`
- Deploy model using Streamlit or Flask

---

## ✍️ Author

- **Bakhshish Sethi**  
  Undergraduate CS Student | ML Enthusiast  
