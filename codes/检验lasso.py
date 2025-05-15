
import json
import os
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]

# 1. 加载数据
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/1906_1909_1911-1916_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['F'] = df['gender'].map({'Male': 0, 'Female': 1})

# 2. 文本向量化
vectorizer = CountVectorizer(binary=True, lowercase=True, min_df=1,
                             tokenizer=custom_tokenizer, stop_words='english')
X = vectorizer.fit_transform(df['comment'])
y = df['F']

# 3. 划分训练／测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. LassoCV 寻找最优 alpha
lasso_cv = LassoCV(cv=5, max_iter=10000, n_alphas=50)
lasso_cv.fit(X_train, y_train)
best_alpha = lasso_cv.alpha_
print(f"Best alpha from CV: {best_alpha:.5f}")

# 可视化 CV MSE 随 alpha 变化
mse_means = np.mean(lasso_cv.mse_path_, axis=1)
plt.figure(figsize=(6,4))
plt.plot(lasso_cv.alphas_, mse_means, marker='o')
plt.axvline(best_alpha, linestyle='--', label=f'alpha={best_alpha:.4f}')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean CV-MSE')
plt.title('CV MSE vs alpha')
plt.legend()
plt.tight_layout()
plt.savefig('cv_mse_vs_alpha.png', dpi=300)
plt.close()
print("已保存：cv_mse_vs_alpha.png")

# 5. 用最优 alpha 重训 Lasso
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train, y_train)

# 6. 训练/测试误差与 R²
y_train_pred = lasso.predict(X_train)
y_test_pred  = lasso.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test  = mean_squared_error(y_test,  y_test_pred)
r2_train  = r2_score(y_train, y_train_pred)
r2_test   = r2_score(y_test,  y_test_pred)

print(f"Train MSE: {mse_train:.4f},  Train R²: {r2_train:.4f}")
print(f"Test  MSE: {mse_test:.4f},  Test  R²: {r2_test:.4f}")

# 7. 调整后 R²
n_train, p = X_train.shape
df_model = np.sum(lasso.coef_ != 0)  # 非零系数个数
r2_adj = 1 - (1 - r2_train) * (n_train - 1) / (n_train - df_model - 1)
print(f"Adjusted R² (train): {r2_adj:.4f} (df_model={df_model})")

# 8. AIC / BIC 近似计算
rss = np.sum((y_train - y_train_pred) ** 2)
aic = 2 * df_model + n_train * np.log(rss / n_train)
bic = np.log(n_train) * df_model + n_train * np.log(rss / n_train)
print(f"AIC: {aic:.2f},  BIC: {bic:.2f}")

# 9. 残差–预测值图
residuals = y_test - y_test_pred
plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, residuals, s=20)
plt.axhline(0, linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (Test set)')
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png', dpi=300)
plt.close()
print("已保存：residuals_vs_predicted.png")

print("Feature matrix shape:", X.shape)
