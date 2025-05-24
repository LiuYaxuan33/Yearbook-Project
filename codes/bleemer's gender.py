import json
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

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


# 计算每个样本的预测值（概率）
y_pred = lasso.predict(X)

# 构建输出 DataFrame
output_df = pd.DataFrame({
    'name': df['name'],
    'gender': df['gender'],
    'predicted_gender_score': y_pred  # 越接近1越倾向于预测为女性，越接近0越倾向于男性
})


mean_pred = np.mean(y_pred)
std_pred = np.std(y_pred)

# 2. 标准化
standardized_pred = (y_pred - mean_pred) / std_pred

# 3. 更新输出 DataFrame
output_df['standardized_pred'] = standardized_pred

# 4. 合并专业信息
# 找出所有以 "is_" 开头的列（这些是专业 dummy）
major_cols = [col for col in df.columns if col.startswith('is_')]

# 将专业信息加入输出 DataFrame
output_df = pd.concat([output_df, df[major_cols]], axis=1)

# 5. 为每个样本标注专业名称（只取第一个为1的 dummy 变量）
def get_major(row):
    for col in major_cols:
        if row[col] == 1:
            return col.replace('is_', '')  # 去掉前缀
    return 'Unknown'

output_df['major'] = output_df.apply(get_major, axis=1)

# 1. 统计：按专业和性别分组，计算均值和标准误
grouped_stats = (
    output_df
    .groupby(['major', 'gender'])['standardized_pred']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
grouped_stats['se'] = grouped_stats['std'] / np.sqrt(grouped_stats['count'])


# 2. 图形绘制：条形图，一上一下表示不同性别

# 为绘图做准备：变成专业 × 性别的多层结构
pivot_df = grouped_stats.pivot(index='major', columns='gender', values='mean')
pivot_se = grouped_stats.pivot(index='major', columns='gender', values='se')
# 获取样本数，用于图中标注
pivot_count = grouped_stats.pivot(index='major', columns='gender', values='count')
pivot_count = pivot_count.fillna(0)

# 专业排序（仍然按女性平均值）
sorted_majors = pivot_df['Female'].sort_values().index.tolist()

x = np.arange(len(sorted_majors))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# 女生柱子（向上）
female_vals = pivot_df.loc[sorted_majors, 'Female']
female_se = pivot_se.loc[sorted_majors, 'Female']
female_n = pivot_count.loc[sorted_majors, 'Female']

bars1 = ax.bar(x, female_vals, width=width, color='lightcoral', label='Female',
               yerr=female_se, capsize=5)

# 男生柱子（向下）
male_vals = pivot_df.loc[sorted_majors, 'Male']
male_se = pivot_se.loc[sorted_majors, 'Male']
male_n = pivot_count.loc[sorted_majors, 'Male']

bars2 = ax.bar(x, -male_vals, width=width, color='skyblue', label='Male',
               yerr=male_se, capsize=5)

# 添加样本数标注
for i in range(len(x)):
    # 女生柱子上方
    ax.text(x[i]+0.2, female_vals[i], f"n={int(female_n[i])}",
            ha='center', va='bottom', fontsize=8)
    
    # 男生柱子下方
    ax.text(x[i]+0.2, -male_vals[i], f"n={int(male_n[i])}",
            ha='center', va='top', fontsize=8)

# 图表样式
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sorted_majors, rotation=45, ha='right')
ax.set_ylabel("Standardized Prediction (z-score)")
ax.set_title("Standardized Lasso Gender Prediction by Major and Gender")
ax.legend()
plt.tight_layout()
plt.savefig("output_/output_bleemer/genderness in majors-mean.png")
plt.show()