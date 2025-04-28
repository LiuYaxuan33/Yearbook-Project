import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')

# 1️⃣ 加载数据
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

comments = [entry["comment"].lower() for entry in data]
names = [entry["name"] for entry in data]
genders = [entry["gender"] for entry in data]

# 2️⃣ 加载 NRC Emotion Lexicon（二值型）
def load_nrc_emotions(path):
    emotion_lexicon = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, emotion, assoc = line.strip().split('\t')
            if int(assoc) == 1:
                emotion_lexicon[word].add(emotion)
    return emotion_lexicon

nrc_lexicon = load_nrc_emotions('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
all_emotion_words = set(nrc_lexicon.keys())
all_emotions = sorted({emotion for emotions in nrc_lexicon.values() for emotion in emotions})

# 3️⃣ 创建 TF-IDF 模型（不清理停用词，保留所有词）
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(comments)
vocab = vectorizer.get_feature_names_out()

# 4️⃣ 计算每条评语在每个情绪维度的 TF-IDF 加权得分
emotion_scores = []

for i, comment_vector in enumerate(tfidf_matrix):
    tfidf_scores = dict(zip(vocab, comment_vector.toarray()[0]))
    emotion_sum = defaultdict(float)

    for word, score in tfidf_scores.items():
        if word in nrc_lexicon:
            for emotion in nrc_lexicon[word]:
                emotion_sum[emotion] += score

    emotion_record = {emo: emotion_sum[emo] for emo in all_emotions}
    emotion_record["name"] = names[i]
    emotion_record["gender"] = genders[i]
    emotion_scores.append(emotion_record)

# 5️⃣ 转成 DataFrame
df_emotions = pd.DataFrame(emotion_scores).fillna(0)
df_emotions.set_index("name", inplace=True)

# 6️⃣ 分性别汇总平均
emotion_columns = [col for col in df_emotions.columns if col != "gender"]
gender_emotion_avg = df_emotions.groupby("gender")[emotion_columns].mean()

# 7️⃣ 绘制雷达图函数（与之前一样）
def plot_radar(df, title):
    labels = df.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    for idx, (label, row) in enumerate(df.iterrows()):
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

# 8️⃣ 绘图
plot_radar(gender_emotion_avg, "TF-IDF Weighted Emotion Comparison by Gender")
