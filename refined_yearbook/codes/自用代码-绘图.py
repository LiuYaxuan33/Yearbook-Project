import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from empath import Empath

# 保存路径
save_path = "empath_emotion_scores.csv"
#save_path = "deepseek_emotion_scores.csv"

# 阈值设置：只保留平均分 >= threshold 的维度
threshold = 0.004

empath_categories = [
    "achievement", "work", 
    "positive_emotion", "negative_emotion", "affection", "trust",
    "independence", "help"
]

df_saved = pd.read_csv(save_path, index_col="name")

# 动态获取所有情感/主题列（排除 metadata）
emotion_cols = [c for c in df_saved.columns if c not in ['gender']]

# 按性别汇总
gender_summary = df_saved.groupby("gender")[emotion_cols].mean()
#gender_summary = df_saved.groupby("gender")[empath_categories].mean()

# 筛选：保留平均分 >= 阈值的维度
filtered_cols = gender_summary.columns[(gender_summary >= threshold).any(axis=0)].tolist()
gender_summary = gender_summary[filtered_cols]

# 输出汇总表
print(f"Empath dimensions with mean >= {threshold} by gender:\n", gender_summary)

# 绘制柱状图：性别维度对比
gender_summary.T.plot(kind='bar', figsize=(12,6), title=f'Empath Dimensions by Gender (Threshold ≥ {threshold})')
plt.ylabel("Normalized Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 雷达图函数
def plot_radar(df_grouped, title):
    labels = df_grouped.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    for idx, row in df_grouped.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# 绘制雷达图（仅满足阈值的维度）
plot_radar(gender_summary, f"Empath Dimensions by Gender (Threshold ≥ {threshold})")
