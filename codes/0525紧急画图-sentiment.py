
from empath import Empath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf

# —— 可切换：是否使用所有 Empath 类别 ——
USE_ALL_CATEGORIES = False  # True: 全量类别, False: 自定义子集
with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

CUSTOM_CATEGORIES = []

for key, value in framework.items():
    CUSTOM_CATEGORIES.extend(value)
father = 'Activities_Engagement'  # 选择的父类
CUSTOM_CATEGORIES = dimension_all[0][father]
SAVE_PATH = 'output_/output_sentiment_tfidf/tfidf.csv'

df = pd.read_csv(SAVE_PATH, index_col=0)

# 6️⃣ 按性别计算平均分
emotion_cols = [c for c in df.columns if c in CUSTOM_CATEGORIES]
gender_summary = df.groupby('gender')[emotion_cols].mean()
print(gender_summary)

# 7️⃣ 可视化：条形图
def plot_bar(df_grouped, title):
    df_grouped.plot(
        kind='bar', figsize=(12,6),
        title=title, fontsize=12
    )
    plt.legend(fontsize=18)
    plt.ylabel('TF-IDF Weighted Sum')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'output_/output_sentiment_tfidf/{father}_tfidf.png', dpi=300, bbox_inches='tight')

plot_bar(gender_summary, father)




