import json
import os
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ 参数配置 ============
JSON_PATH = 'all_data_use_labeled.json'
OUTPUT_DIR = 'output'
B = 10  # Placebo 随机实验次数
TAU_PCT = 95  # 阈值分位数
CV_FOLDS = 5  # LassoCV 折数
RANDOM_STATE = 42

# ============ 初始化 ============
os.makedirs(OUTPUT_DIR, exist_ok=True)
nlp = spacy.load("en_core_web_sm")

# 自定义分词，只保留 ADJ 和 ADV
def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]

# ============ 1. 加载原始标签与创建 Placebo 标签 ============
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 提取真实标签列（例如 MAJOR_TO_ANALYZE 指定的专业）
df['label_true'] = df['gender'].map({'Male': 0, 'Female': 1})

# 随机打乱顺序（仅打乱数据行，保留原标签）
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 创建 Placebo 标签，使用 iloc 避免 off-by-one
half = int(df['label_true'].sum())
df['label_placebo'] = 0
df.iloc[:half, df.columns.get_loc('label_placebo')] = 1

y_true = df['label_true'].values
ny_placebo = df['label_placebo'].values

# ============ 2. 文本向量化 ============
vectorizer = CountVectorizer(
    binary=True,
    lowercase=True,
    min_df=1,
    tokenizer=custom_tokenizer,
    stop_words='english'
)
X = vectorizer.fit_transform(df['comment'])  # (n_samples, n_features)
print(f"Feature matrix shape: {X.shape}")

# ============ 3. Lasso 训练函数 ============
def run_lasso(X, y, cv=CV_FOLDS, random_state=RANDOM_STATE):
    """使用 LassoCV 自动选 alpha，返回系数和最佳 alpha"""
    lasso = Lasso(alpha = 0.0008, max_iter=10000)
    lasso.fit(X, y)
    return lasso.coef_

# ============ 4. Placebo 实验 ============
n_samples, n_features = X.shape
coef_placebo = np.zeros((B, n_features))
for b in tqdm(range(B), desc="Placebo Lasso runs"):
    # 随机打乱标签
    y_b = np.random.permutation(ny_placebo)
    coef_b= run_lasso(X, y_b)
    coef_placebo[b, :] = coef_b

# ============ 5. 计算阈值 tau ============
tau = np.percentile(np.abs(coef_placebo), TAU_PCT, axis=0)


# ============ 6. 真标签 Lasso ============
coef_true = run_lasso(X, y_true)

# ============ 7. 筛选超阈值特征 ============
selected_idx = np.where(np.abs(coef_true) > tau)[0]
print(f"超出 Placebo 阈值的特征数量: {len(selected_idx)}")

# ============ 8. 输出结果表格 ============
features = vectorizer.get_feature_names_out()
results_df = pd.DataFrame({
    'feature': features,
    'coef_true': coef_true,
    'tau_placebo': tau
})
results_df['selected'] = np.abs(results_df['coef_true']) > results_df['tau_placebo']
results_df.to_csv(os.path.join(OUTPUT_DIR, 'lasso_placebo_results.csv'), index=False)
results_df[results_df['selected']].sort_values(by='coef_true', key=lambda s: np.abs(s), ascending=False)  .to_csv(os.path.join(OUTPUT_DIR, 'selected_features.csv'), index=False)

# ============ 9. 可视化示例 ============
if len(selected_idx) > 0:
    j = selected_idx[0]  # 示例第一个被选特征
    plt.figure(figsize=(8, 4))
    plt.hist(coef_placebo[:, j], bins=30, density=True)
    plt.axvline(coef_true[j], color='red', linestyle='--',
                label=f'True coef={coef_true[j]:.3f}')
    plt.axvline(-tau[j], color='gray', linestyle=':', label=f'±{TAU_PCT}%阈值')
    plt.axvline(tau[j], color='gray', linestyle=':')
    plt.title(f"feature '{features[j]}' Placebo vs True coef comparison")
    plt.xlabel('Coefficient')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'coef_compare_{features[j]}.png'), dpi=300)
    plt.close()
    print(f"可视化已保存：{os.path.join(OUTPUT_DIR, f'coef_compare_{features[j]}.png')}")
else:
    print("无特征超出阈值，无可视化展示。")

print("Placebo 检验全流程完成。所有输出保存在 output 文件夹中。")
