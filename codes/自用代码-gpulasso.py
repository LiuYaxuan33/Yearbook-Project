import json
import os
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from cuml.linear_model import Lasso as cuLasso
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# ⚙️ 加载 GPU 模型
nlp = spacy.load("en_core_web_trf")  # 使用 Transformer 模型，支持 GPU
BLACKLIST = {}
USE_1GRAM = False
MIN_DF = 5

def generate_features(texts):
    for doc in nlp.pipe(texts, batch_size=50):
        # 先构造一个布尔数组，标记每个 token 是否“有效”
        valid_mask = [
            (not token.is_stop
             and not token.is_punct
             and token.text.lower() not in BLACKLIST)
            for token in doc
        ]
        
        # 构造 tokens_lower 与 pos_tags 两个列表，仅用于 1-gram
        tokens_lower = [token.text.lower() for token, valid in zip(doc, valid_mask) if valid]
        pos_tags =       [token.pos_       for token, valid in zip(doc, valid_mask) if valid]
        
        # 第一步：生成1-gram（仅 ADJ/ADV）
        valid_terms = []
        if USE_1GRAM:
            for tok, pos in zip(tokens_lower, pos_tags):
                if pos in {'ADJ', 'ADV'}:
                    valid_terms.append(tok)
        
        # 第二步：生成2-gram
        # 基于原始 doc 中的相邻 token 对
        for i in range(len(doc) - 1):
            t1, t2 = doc[i], doc[i+1]
            # 两个词必须都是“有效”的
            if not (valid_mask[i] and valid_mask[i+1]):
                continue
            # 至少一个是 ADJ/ADV/NOUN
            if t1.pos_ in {'ADJ', 'ADV', 'NOUN'} or t2.pos_ in {'ADJ', 'ADV', 'NOUN'}:
                bigram = f"{t1.text.lower()}_{t2.text.lower()}"
                valid_terms.append(bigram)
        
        yield ' '.join(valid_terms)

# 加载数据
with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['F'] = df['gender'].map({'Male': 1, 'Female': 0})

# 文本处理
df['processed'] = list(generate_features(df['comment']))

# 向量化
vectorizer = CountVectorizer(
    binary=True,
    tokenizer=lambda x: x.split(),
    lowercase=False,
    min_df=MIN_DF
)
X = vectorizer.fit_transform(df['processed'])
y = df['F'].values.astype(np.float32)  # cuml 需要 float32 输入
X = X.astype(np.float32)

# Lasso 回归（GPU）
print("开始 GPU Lasso 回归...")
lasso = cuLasso(alpha=0.01, max_iter=10000)  # 你可以调节 alpha
lasso.fit(X, y)

# 提取非零系数
coefs = lasso.coef_.toarray().flatten()
selected_features = np.where(coefs != 0)[0]
vocab_selected = vectorizer.get_feature_names_out()[selected_features]
coefs_selected = coefs[selected_features]

# 保存结果
result_df = pd.DataFrame({
    'word': vocab_selected,
    'coefficient': coefs_selected
}).sort_values('coefficient', ascending=False)
result_df.to_csv("new_2-grams_gpu.csv", index=False)

# 控制台打印
print("\n非零系数特征列表：")
print(result_df.to_string(index=False))

# 可视化
df_sorted = result_df.sort_values('coefficient')
n_items = len(df_sorted)
fig_height = max(8, n_items * 0.3)
os.makedirs('./figures', exist_ok=True)
save_path = f'./figures/lasso_gpu_coeffs_alpha_0.01.png'

plt.figure(figsize=(8, fig_height))
plt.scatter(df_sorted['coefficient'], df_sorted['word'], s=50)
plt.axvline(x=0, linestyle='--')
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.xlabel('Lasso Coefficients (GPU)', fontsize=12)
plt.ylabel('Words', fontsize=12)
plt.title(f'Figure 1 (alpha=0.01)', fontsize=14)
plt.yticks(fontsize=8)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'图已保存到：{save_path}')
