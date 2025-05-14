
from empath import Empath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# —— 可切换：是否使用所有 Empath 类别 ——
USE_ALL_CATEGORIES = True  # True: 全量类别, False: 自定义子集
CUSTOM_CATEGORIES = [
    "ability","grindstone","research","standout","teaching","Citizenship","Recuitment Prospects"
]
NONE_CATEGORIES = [
    "ability","grindstone","research","standout","teaching","Citizenship","Recuitment Prospects"
]


# 文件路径
DATA_PATH = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/all_data_use.json'
SAVE_PATH = 'empath_tfidf_emotion_scores.csv'

# 1️⃣ 读取数据
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

comments = [entry['comment'].lower() for entry in data]
names    = [entry['name'] for entry in data]
genders  = [entry['gender'] for entry in data]

# 2️⃣ Empath 初始化
lexicon = Empath()
all_categories = list(lexicon.cats.keys())
if USE_ALL_CATEGORIES:
    empath_categories = all_categories
else:
    empath_categories = CUSTOM_CATEGORIES

# 构建类别—词集合映射
category_words = {cat: set(lexicon.cats.get(cat, [])) for cat in empath_categories}

# 3️⃣ 构建 TF-IDF 矩阵
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(comments)
vocab        = vectorizer.get_feature_names_out()

# 4️⃣ 计算 TF-IDF 加权 Empath 得分
results = []
for idx, entry in enumerate(data):
    name   = entry['name']
    gender = entry['gender']
    tfidf_scores = dict(zip(vocab, tfidf_matrix[idx].toarray()[0]))

    scores = {}
    for cat in empath_categories:
        # 累加类别中所有词的 TF-IDF 值
        scores[f'{cat}_tfidf'] = sum(
            tfidf_scores.get(word, 0.0) for word in category_words[cat]
        )
    scores['name']   = name
    scores['gender'] = gender
    results.append(scores)

# 5️⃣ DataFrame
df = pd.DataFrame(results).set_index('name')
# 保存至 CSV
df.to_csv(SAVE_PATH)

# 6️⃣ 按性别计算平均分
emotion_cols = [c for c in df.columns if c.endswith('_tfidf')]
gender_summary = df.groupby('gender')[emotion_cols].mean()
print(gender_summary)

# 7️⃣ 可视化：条形图
gender_summary.T.plot(
    kind='bar', figsize=(12,6),
    title='Empath TF-IDF Weighted Dimensions by Gender'
)
plt.ylabel('TF-IDF Weighted Sum')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8️⃣ 可视化：雷达图
def plot_radar(df_grouped, title):
    labels = df_grouped.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(polar=True))
    for label, row in df_grouped.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    plt.tight_layout()
    plt.show()

plot_radar(gender_summary, 'Empath TF-IDF Weighted Dimensions by Gender')

