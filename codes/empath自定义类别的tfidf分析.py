
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

EJ_REFERENCE_CATEGORIES = [
    "ability","grindstone","research","standout","teaching&Citizenship","Recuitment Prospects"
]

MODEL = "fiction"

ALL_LABELS = [
    "is_agriculture",
    "is_home economics",
    "is_science",
    "is_engineering",
    "is_music",
    "is_education",
    "is_veterinary"]

# 文件路径
DATA_PATH = 'all_data_use_labeled.json'
SAVE_PATH = 'output_/output_sentiment_tfidf/tfidf.csv'

# 1️⃣ 读取数据
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

comments = [entry['comment'].lower() for entry in data]


# 2️⃣ Empath 初始化
lexicon = Empath()
all_categories = list(lexicon.cats.keys())
if USE_ALL_CATEGORIES:
    empath_categories = all_categories
else:
    empath_categories = CUSTOM_CATEGORIES
for category in CUSTOM_CATEGORIES:
    if category not in all_categories:
        lexicon.create_category(category, NEW_WORDS[category], model = MODEL)

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
        scores[cat] = sum(
            tfidf_scores.get(word, 0.0) for word in category_words[cat]
        )
    scores['name']   = name
    scores['gender'] = gender
    for labels in ALL_LABELS:
        if entry[labels] == 1:
            scores[labels.replace(" ", "_")] = 1
        else:
            scores[labels.replace(" ", "_")] = 0
    if "Iowa" in entry['hometown']:
        scores['hometown_Iowa'] = 1
    else:
        scores['hometown_Iowa'] = 0
    for key, value in framework.items():
        sum_value = 0
        for cat in value:
            sum_value += scores[cat]
        scores[key + "_mean"] = sum_value/len(value)
    # 计算 Empath 参考类别的 TF-IDF 加权得分
    results.append(scores)

# 5️⃣ DataFrame
df = pd.DataFrame(results).set_index('name')
# 保存至 CSV
df.to_csv(SAVE_PATH)

# 6️⃣ 按性别计算平均分
emotion_cols = [c for c in df.columns if c in CUSTOM_CATEGORIES]
gender_summary = df.groupby('gender')[emotion_cols].mean()
print(gender_summary)

# 7️⃣ 可视化：条形图
def plot_bar(df_grouped, title):
    df_grouped.plot(
        kind='bar', figsize=(12,6),
        title=title
    )
    plt.ylabel('TF-IDF Weighted Sum')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#plot_bar(gender_summary, 'Empath TF-IDF Weighted Dimensions by Gender')

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

#plot_radar(gender_summary, 'Empath TF-IDF Weighted Dimensions by Gender')


# ✅ 构建自变量列表
independent_vars = ["gender"]

# ✅ 因变量（子类别得分 + 大类别均值得分）
emotion_cols = CUSTOM_CATEGORIES
major_cat_cols = [f"{cat}_mean" for cat in framework.keys()]

# ✅ 拟合 OLS 回归并保存结果
ols_results = {}

for target in emotion_cols + major_cat_cols:
    formula = f"{target} ~ " + " + ".join(independent_vars)
    model = smf.ols(formula, data=df).fit()
    ols_results[target] = model

summary_data = []

for target, model in ols_results.items():
    for var in model.params.index:
        coef = model.params[var]
        std_err = model.bse[var]
        p_val = model.pvalues[var]

        if p_val < 0.01:
            significance = '***'
        elif p_val < 0.05:
            significance = '**'
        elif p_val < 0.1:
            significance = '*'
        else:
            significance = ''

        summary_data.append({
            'Dependent Variable': target,
            'Variable': var,
            'Coefficient': coef,
            'Std. Error': std_err,
            'p-value': p_val,
            'Significance': significance
        })

regression_summary_df = pd.DataFrame(summary_data)


# 导出为 CSV
regression_summary_df.to_csv("output_/output_sentiment_tfidf/no_control_tfidf-regression.csv", index=False)

