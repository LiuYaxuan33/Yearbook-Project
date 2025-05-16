import json
import os
import spacy
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_trf")

def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]

# 1. 读取数据
with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['F'] = df['gender'].map({'Male': 0, 'Female': 1})

# 2. 文本特征
vectorizer = CountVectorizer(
    binary=True, lowercase=True, min_df=5,
    tokenizer=custom_tokenizer, stop_words='english'
)
X_text = vectorizer.fit_transform(df['comment'])
p_text = X_text.shape[1]

# 3. major 的 one-hot 哑变量
maj_dummies = pd.get_dummies(df['major'], prefix='maj')
X_maj = sparse.csr_matrix(maj_dummies.values)
p_maj = X_maj.shape[1]

# 4. （第一步）OLS 拟合 major 部分，算残差 r = y - X_maj β_maj
y = df['F'].values
ols = LinearRegression(fit_intercept=True)
ols.fit(X_maj.toarray(), y)          # 小规模的 dummy 矩阵，toarray() 通常没问题
beta_maj = ols.coef_               # shape (p_maj,)
r = y - ols.predict(X_maj.toarray())

# 5. （第二步）对残差 r 做 LassoCV
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_text, r)
best_alpha = lasso_cv.alpha_

# 用最优 alpha 训练 final Lasso
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_text, r)
beta_text = lasso.coef_           # shape (p_text,)

# 6. 收集非零系数
#   文本特征
idx_text = np.where(beta_text != 0)[0]
words_sel = vectorizer.get_feature_names_out()[idx_text]
coefs_text = beta_text[idx_text]

#   major 特征（OLS 部分，理论上所有都可能非零）
idx_maj = np.where(beta_maj != 0)[0]
maj_names = maj_dummies.columns[idx_maj]
coefs_maj = beta_maj[idx_maj]

# 7. 合并结果 & 输出
result = pd.DataFrame({
    'feature': np.concatenate([words_sel, maj_names]),
    'coefficient': np.concatenate([coefs_text, coefs_maj])
}).sort_values('coefficient', ascending=False)

result.to_csv("lasso_selected_features_with_major.csv", index=False)
print(result)

# 8. 绘图（同你原来逻辑，只要从 result 取值即可）
df_sorted = result.sort_values('coefficient')
n_items = len(df_sorted)
fig_h = max(8, n_items * 0.3)
plt.figure(figsize=(8, fig_h))
plt.scatter(df_sorted['coefficient'], df_sorted['feature'], s=50)
plt.axvline(x=0, linestyle='--')
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title(f'Lasso on Text Residuals (OLS majors), alpha={best_alpha:.4f}')
plt.yticks(fontsize=8)
out_png = f'lasso_coeffs_text_resid_alpha_{best_alpha:.4f}.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f'图已保存到：{out_png}')

print("文本特征矩阵形状:", X_text.shape)
print("major 矩阵形状:", X_maj.shape)
