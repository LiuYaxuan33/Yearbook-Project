import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 下载英文分词器
nltk.download('punkt')

# 1️⃣ 读取 JSON 文件（假设名为 students.json）
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2️⃣ 载入 NRC Emotion Lexicon
def load_nrc_lexicon(filepath):
    lexicon = defaultdict(set)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, emotion, assoc = line.strip().split('\t')
            if int(assoc) == 1:
                lexicon[word].add(emotion)
    return lexicon

nrc_lexicon = load_nrc_lexicon('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')  # 修改为你的路径

# 3️⃣ 情绪分析函数
def analyze_emotions(comment, lexicon):
    tokens = word_tokenize(comment.lower())
    emotion_count = defaultdict(int)

    for word in tokens:
        if word in lexicon:
            for emotion in lexicon[word]:
                emotion_count[emotion] += 1

    return dict(emotion_count)

# 4️⃣ 对所有学生进行分析
results = []

for student in data:
    comment = student.get("comment", "")
    name = student.get("name", "")
    emotion_counts = analyze_emotions(comment, nrc_lexicon)
    emotion_counts["name"] = name
    emotion_counts["year"] = student.get("year", "")
    emotion_counts["gender"] = student.get("gender", "")
    emotion_counts["major"] = student.get("major", "")
    results.append(emotion_counts)

# 5️⃣ 转换为 DataFrame
emotion_df = pd.DataFrame(results).fillna(0)
emotion_df.set_index("name", inplace=True)

# 6️⃣ 保存结果（可选）
emotion_df.to_csv("student_emotion_analysis.csv")

# 7️⃣ 显示结果
print(emotion_df.head())



# 1️⃣ 提取情绪维度字段（排除非情绪字段）
emotion_columns = [col for col in emotion_df.columns if col not in ['gender', 'year', 'major']]

# 2️⃣ 按性别计算情绪均值
gender_grouped = emotion_df.groupby("gender")[emotion_columns].mean()

# 3️⃣ 统一雷达图绘制函数
def plot_radar(df, title):
    labels = df.columns.tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for idx, (label, row) in enumerate(df.iterrows()):
        values = row.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# 4️⃣ 绘制性别对比情绪雷达图
plot_radar(gender_grouped, "Gender-based Emotion Distribution in Graduation Comments")
