from empath import Empath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os

save_path = "empath_emotion_scores.csv"

# 📂 读取 students.json 文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 🔍 分析维度
empath_categories = [
    "achievement", "work",
    "positive_emotion", "negative_emotion", "affection", "trust",
    "independence", "help"
]


lexicon = Empath()
results = []
print(1)
# 🎯 对每条评论分析 Empath 维度
for student in data:
    comment = student["comment"].lower()
    gender = student["gender"]
    name = student["name"]
    #scores = lexicon.analyze(comment, categories=empath_categories, normalize=True)
    scores = lexicon.analyze(comment, normalize=True)
    scores["gender"] = gender
    scores["name"] = name
    results.append(scores)
print(2)
# 📊 转为 DataFrame
df_empath = pd.DataFrame(results)
df_empath.set_index("name", inplace=True)
print(3)
# 📈 性别平均维度分数
gender_summary = df_empath.groupby("gender")[empath_categories].mean()
# ✅ 输出
print(gender_summary)


df_empath.to_csv(save_path, mode="a", index=True, header=True)
# 📊 性别维度对比条形图
gender_summary.T.plot(kind='bar', figsize=(12,6), title='Empath Dimensions by Gender')
plt.ylabel("Normalized Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df = pd.read_csv(save_path)
meta_cols = ['name', 'gender', 'year']
emotion_cols = [col for col in df.columns if col not in meta_cols]
gender_grouped = df.groupby("gender")[emotion_cols].mean()

def plot_radar(df_grouped, title):
    labels = df_grouped.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for idx, (label, row) in enumerate(df_grouped.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# 📈 绘图
plot_radar(gender_grouped, "Empath Dimensions by Gender")