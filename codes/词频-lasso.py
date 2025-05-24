import json
import os
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]  # 形容词 (ADJ) 和副词 (ADV)

# 1. 从 JSON 文件中加载数据
with open('all_data_use_labeled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. 将性别转换为二元变量：Female=1, Male=0
df['F'] = df['gender'].map({'Male': 0, 'Female': 1})

# 3. 提取评论文本，并利用 CountVectorizer 构建二值词汇矩阵
#vectorizer = CountVectorizer(binary=True, lowercase=True, min_df=1, tokenizer=custom_tokenizer)
vectorizer = CountVectorizer(binary=True, lowercase=True, min_df=1, tokenizer=custom_tokenizer, stop_words='english')
X = vectorizer.fit_transform(df['comment'])
y = df['F']

# 4. 使用 Lasso 回归（带交叉验证选择正则化系数）
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X, y)

# 用最佳 alpha 重新训练模型
best_alpha = lasso_cv.alpha_
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X, y)

coefs = lasso.coef_
selected_features = np.where(coefs != 0)[0]
vocab_selected = vectorizer.get_feature_names_out()[selected_features]
coefs_selected = coefs[selected_features]

# 创建完整的结果表格并按系数排序
result_df = pd.DataFrame({
    'word': vocab_selected,
    'coefficient': coefs_selected
}).sort_values('coefficient', ascending=False)  # 按系数从大到小排序

# 输出表格到CSV文件
result_df.to_csv("output_/output_basic/1-grams.csv", index=False)

# 控制台打印结果
print("\n非零系数特征列表：")
print(result_df.to_string(index=False))




# 整理数据用于绘图
plot_df = pd.DataFrame({
    'word': vocab_selected,
    'coef': coefs_selected
})


# 5. 绘制回归系数图（Lasso 无标准误信息）

# 1. 按系数排序
df_sorted = result_df.sort_values('coefficient')

# 2. 动态计算图高
n_items = len(df_sorted)
fig_height = max(8, n_items * 0.3)

save_dir = './output_/output_basic/'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'1-grams.png')


# 3. 绘图
plt.figure(figsize=(8, fig_height))
plt.scatter(df_sorted['coefficient'], df_sorted['word'], s=50)
plt.axvline(x=0, linestyle='--')
plt.grid(axis='x', linestyle='--', linewidth=0.5)

# 4. 标注与样式
plt.xlabel('Lasso Regression Coefficients', fontsize=12)
plt.ylabel('Words', fontsize=12)
plt.title(f'Figure 1 (Best alpha={best_alpha:.4f})', fontsize=14)
plt.yticks(fontsize=8)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'图已保存到：{save_path}')

# 输出稀疏矩阵维度
print("Feature matrix shape:", X.shape)