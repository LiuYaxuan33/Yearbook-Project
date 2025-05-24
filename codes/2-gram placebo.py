import json
import os
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import glob

# ============ 参数配置 ============
JSON_PATH = 'all_data_use_labeled.json'
OUTPUT_DIR = 'output'
PLACEBO_CACHE_DIR = os.path.join(OUTPUT_DIR, 'placebo_cache')
B = 10000  # Placebo 实验次数
TAU_PCT = 95  # 阈值分位数
RANDOM_STATE = 42

# ============ 初始化 ============
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLACEBO_CACHE_DIR, exist_ok=True)
nlp = spacy.load("en_core_web_sm")
BLACKLIST = {}  # 自定义黑名单词
USE_1GRAM = False
MIN_DF = 5  # 最小文档频率阈值

def generate_features(texts):
    for doc in nlp.pipe(texts, batch_size=50):
        # 先构造一个布尔数组，标记每个 token 是否“有效”
        valid_mask = [
            (not token.is_punct
             and not token.is_stop
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

# 自定义分词器，只保留 ADJ 和 ADV
def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]

# ============ 1. 加载原始标签与 Placebo 标签 ============
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 提取真实标签
df['label_true'] = df['gender'].map({'Male': 0, 'Female': 1})
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 创建 Placebo 标签
half = int(df['label_true'].sum())
df['label_placebo'] = 0
df.iloc[:half, df.columns.get_loc('label_placebo')] = 1

y_true = df['label_true'].values
ny_placebo = df['label_placebo'].values

# ============ 2. 文本向量化 ============
vectorizer = CountVectorizer(
    binary=True,
    tokenizer=lambda x: x.split(),  # 按空格分割预处理后的字符串
    lowercase=False,  # 预处理阶段已统一小写
    min_df=MIN_DF
)
df['processed'] = list(generate_features(df['comment']))
X = vectorizer.fit_transform(df['processed'])
print(f"Feature matrix shape: {X.shape}")

# ============ 3. Lasso 回归函数 ============
def run_lasso(X, y):
    lasso = Lasso(alpha=0.0008, max_iter=10000)
    lasso.fit(X, y)
    return lasso.coef_

# ============ 4. Placebo 并行任务 ============
def run_and_save_placebo(b, X, y_placebo, save_dir):
    save_path = os.path.join(save_dir, f'coef_placebo_{b}.npy')
    if os.path.exists(save_path):
        return
    y_b = np.random.permutation(y_placebo)
    coef_b = run_lasso(X, y_b)
    np.save(save_path, coef_b)

# 查找已完成任务
existing_files = glob.glob(os.path.join(PLACEBO_CACHE_DIR, 'coef_placebo_*.npy'))
existing_indices = set(int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in existing_files)
remaining_indices = [b for b in range(B) if b not in existing_indices]

print(f"Total placebo B={B}, done={len(existing_indices)}, remaining={len(remaining_indices)}")

# 并行执行剩余任务
Parallel(n_jobs=-1)(
    delayed(run_and_save_placebo)(b, X, ny_placebo, PLACEBO_CACHE_DIR)
    for b in tqdm(remaining_indices, desc="Running placebo Lasso in parallel")
)

# 合并所有 Placebo 结果
n_features = X.shape[1]
coef_placebo = np.zeros((B, n_features))
for b in range(B):
    coef_placebo[b] = np.load(os.path.join(PLACEBO_CACHE_DIR, f'coef_placebo_{b}.npy'))

# ============ 5. 计算阈值 ============
tau = np.percentile(np.abs(coef_placebo), TAU_PCT, axis=0)

# ============ 6. 真标签 Lasso ============
coef_true = run_lasso(X, y_true)

# ============ 7. 筛选重要特征 ============
selected_idx = np.where(np.abs(coef_true) > tau)[0]
print(f"超出 Placebo 阈值的特征数量: {len(selected_idx)}")

# ============ 8. 保存结果表格 ============
features = vectorizer.get_feature_names_out()
results_df = pd.DataFrame({
    'feature': features,
    'coef_true': coef_true,
    'tau_placebo': tau
})
results_df['selected'] = np.abs(results_df['coef_true']) > results_df['tau_placebo']

results_df.to_csv(os.path.join(OUTPUT_DIR, 'lasso_placebo_results.csv'), index=False)
results_df[results_df['selected']] \
    .sort_values(by='coef_true', key=lambda s: np.abs(s), ascending=False) \
    .to_csv(os.path.join(OUTPUT_DIR, 'selected_features.csv'), index=False)

# ============ 9. 可视化示例 ============
if len(selected_idx) > 0:
    # 构建可视化数据
    selected_features_df = results_df[results_df['selected']].copy()
    selected_features_df['abs_coef'] = selected_features_df['coef_true'].abs()
    selected_features_df = selected_features_df.sort_values(by='abs_coef', ascending=True)

    # 可选：限制最多显示前N个（按绝对值排序）
    N_SHOW = 30
    if len(selected_features_df) > N_SHOW:
        selected_features_df = selected_features_df.tail(N_SHOW)

    fig_height = 0.35 * len(selected_features_df)
    plt.figure(figsize=(8, fig_height))

    # 绘制散点图
    plt.scatter(selected_features_df['coef_true'], selected_features_df['feature'], s=60, color='dodgerblue', label='True Coef')

    # 添加阈值区间线（上下）
    plt.axvline(x=0, linestyle='--', color='black')
    plt.axvline(x=+tau.max(), linestyle=':', color='gray', label=f'+{TAU_PCT}th placebo threshold')
    plt.axvline(x=-tau.max(), linestyle=':', color='gray', label=f'-{TAU_PCT}th placebo threshold')

    # 美化样式
    plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel('Lasso Coefficient', fontsize=12)
    plt.ylabel('Feature (adj/adv)', fontsize=12)
    plt.title(f'Placebo-Filtered Features (Total Selected = {len(selected_idx)})', fontsize=14)
    plt.yticks(fontsize=9)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    fig_path = os.path.join(OUTPUT_DIR, 'selected_features_plot.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存：{fig_path}")
else:
    print("无特征超出阈值，无可视化展示。")
     

print("Placebo 检验全流程完成。所有输出保存在 output 文件夹中。")
