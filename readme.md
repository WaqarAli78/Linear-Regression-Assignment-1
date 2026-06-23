# 📊 Model Results: Simple Linear Regression (Age vs Insurance Premium)

After training the Linear Regression model on the dataset, the following results were obtained:

---

## 📁 Dataset Summary

| Feature | Value |
|--------|------|
| Shape  | (7, 2) |
| Features | Age |
| Target | Premium |

---

## 🧠 Model Parameters

The trained model learned the following equation:


Premium = -10114.73 + (1172.95 × Age)


| Parameter | Value |
|----------|------|
| Intercept (β₀) | -10114.7260 |
| Coefficient (β₁) | 1172.9452 |

---

## 📊 Model Performance

| Metric | Value |
|-------|------|
| R² Score | 0.9689 |
| MAE (Mean Absolute Error) | 937.38 |
| MSE (Mean Squared Error) | 1052348.34 |
| RMSE (Root Mean Squared Error) | 1025.84 |

---

## 🎯 Prediction Example

For a new input:


Age = 35


### Predicted Output:


Premium = 30,938.36


---

## 📈 Key Insights

- The model shows a **strong linear relationship** between Age and Premium.
- R² Score of **0.9689** indicates the model explains ~97% of variance.
- Prediction error (RMSE ≈ 1025) is acceptable for such a small dataset.
- Insurance premium increases significantly with age.

---

## 📌 Conclusion

This model successfully demonstrates:

✔ How Linear Regression works in real-world scenarios  
✔ How to interpret coefficients and intercept  
✔ How to evaluate regression performance  
✔ How predictions are generated from learned patterns  

---


---
