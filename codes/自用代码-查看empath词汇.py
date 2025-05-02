
from empath import Empath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# —— 可切换：是否使用所有 Empath 类别 ——
USE_ALL_CATEGORIES = False  # True: 全量类别, False: 自定义子集
CUSTOM_CATEGORIES = [
    "achievement", "work",
    "positive_emotion", "negative_emotion", "affection", "trust",
    "independence", "help"
]

SAVE_PATH = 'empath_tfidf_emotion_scores.csv'


# 2️⃣ Empath 初始化
lexicon = Empath()
all_categories = list(lexicon.cats.keys())
if USE_ALL_CATEGORIES:
    empath_categories = all_categories
else:
    empath_categories = CUSTOM_CATEGORIES

# 构建类别—词集合映射
category_words = {cat: set(lexicon.cats.get(cat, [])) for cat in empath_categories}


# 将 category_words 写入 txt 文件
output_file = "empath_category_words.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for category, words in category_words.items():
        f.write(f"{category}:\n")
        f.write(", ".join(sorted(words)) + "\n\n")

print(f"category_words 已保存到 {output_file}")

