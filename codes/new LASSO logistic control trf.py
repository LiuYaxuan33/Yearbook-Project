import json
import os
import spacy
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LassoCV, Lasso
import matplotlib.pyplot as plt

# --------------- 1. 加载数据 ---------------
with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['F'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(float)

# --------------- 2. 文本特征 ---------------
nlp = spacy.load("en_core_web_sm")
def custom_tokenizer(text):
    doc = nlp(text)
    return [tok.text for tok in doc if tok.pos_ in ('ADJ','ADV')]

vectorizer = CountVectorizer(
    binary=True, lowercase=True,
    tokenizer=custom_tokenizer,
    stop_words='english'
)
X_text = vectorizer.fit_transform(df['comment'])  # (n_samples, p_text)
p_text = X_text.shape[1]

# --------------- 3. major 的 one-hot ---------------
maj_dummies = pd.get_dummies(df['major'], prefix='maj')
X_maj = sparse.csr_matrix(maj_dummies.values)  # (n_samples, p_maj)
p_maj = X_maj.shape[1]

# --------------- 4. 合并原始特征 ---------------
X_orig = sparse.hstack([X_text, X_maj], format='csr')  # (n_samples, p_text+p_maj)
y = df['F'].values

# --------------- 5. 定义每列的“惩罚因子” p_j ---------------
#   文本列 p_j = 1，major 列 p_j = epsilon（很小）
epsilon = 1e-3
penalty_factors = np.concatenate([
    np.ones(p_text, dtype=float),
    np.full(p_maj, epsilon, dtype=float)
])  # 长度 = p_text + p_maj

# --------------- 6. 按每列因子缩放特征 ---------------
#    X_scaled[:, j] = X_orig[:, j] / p_j
#    这里先把 sparse 转成 CSC 方便按列操作（若占内存可分块或逐列）
X_csc = X_orig.tocsc(copy=True)
for j, pj in enumerate(penalty_factors):
    if pj != 1.0:
        X_csc[:, j] = X_csc[:, j] * (1.0 / pj)
X_scaled = X_csc.tocsr()

# --------------- 7. 用 sklearn 的 LassoCV 拟合缩放后的数据 ---------------
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)
best_alpha = lasso_cv.alpha_
print(f"Best α from CV: {best_alpha:.4f}")

# --------------- 8. 重新用最佳 α 训练 Lasso ---------------
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_scaled, y)
beta_scaled = lasso.coef_  # 对缩放后 X 的系数

# --------------- 9. 换算回原始坐标系的系数 ---------------
#    因为      X_scaled[:,j] = X_orig[:,j] / p_j
# => 原始系数 β_j = β_scaled_j / (1/p_j) = β_scaled_j * p_j
beta_orig = beta_scaled * penalty_factors

# --------------- 10. 筛选并输出 ---------------
# 文本特征
idx_text = np.where(beta_orig[:p_text] != 0)[0]
words_sel  = vectorizer.get_feature_names_out()[idx_text]
coefs_text = beta_orig[idx_text]

# major 特征
idx_maj = np.where(beta_orig[p_text:] != 0)[0]
maj_names  = maj_dummies.columns[idx_maj]
coefs_maj  = beta_orig[p_text:][idx_maj]

# 合并结果
result_df = pd.DataFrame({
    'feature': np.concatenate([words_sel, maj_names]),
    'coefficient': np.concatenate([coefs_text, coefs_maj])
}).sort_values('coefficient', ascending=False)

out_csv = "lasso_pf_scaled_selected.csv"
result_df.to_csv(out_csv, index=False)
print(f"Selected features saved to {out_csv}")

# --------------- 11. （可选）绘图 ---------------
df_sorted = result_df.sort_values('coefficient')
n_items = len(df_sorted)
fig_h  = max(8, n_items * 0.3)
plt.figure(figsize=(8, fig_h))
plt.scatter(df_sorted['coefficient'], df_sorted['feature'], s=50)
plt.axvline(x=0, linestyle='--')
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title(f'Lasso with per-feature scaling (α={best_alpha:.4f})', fontsize=14)
plt.yticks(fontsize=8)
png_path = f'lasso_scaled_coeffs_alpha_{best_alpha:.4f}.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Plot saved to: {png_path}')
