
from empath import Empath
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# —— 可切换：是否使用所有 Empath 类别 ——
USE_ALL_CATEGORIES = False  # True: 全量类别, False: 自定义子集

SAVE_PATH = 'empath_tfidf_emotion_scores.csv'

with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

CUSTOM_CATEGORIES = []

for key, value in framework.items():
    CUSTOM_CATEGORIES.extend(value)

MODEL = "fiction"


lexicon = Empath()
lexicon.create_category("intelligence", NEW_WORDS["intelligence"], model=MODEL)