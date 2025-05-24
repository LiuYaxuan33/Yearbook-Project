
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

# 文件路径
DATA_PATH = 'all_data_use_labeled.json'
SAVE_PATH = 'output_/output_dpsk/deepseek.csv'
FINAL_SAVE_PATH = 'output_/output_dpsk/deepseek_final.csv'

# 1️⃣ 读取数据
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
data2 = pd.read_csv(SAVE_PATH, encoding='utf-8')

df = pd.DataFrame(data2)

# 6️⃣ 按性别计算平均分
emotion_cols = [c for c in df.columns if c in CUSTOM_CATEGORIES]
gender_summary = df.groupby('gender')[emotion_cols].mean()
print(gender_summary)

for key, value in framework.items():
    df[key + "_mean"] = df[value].mean(axis=1)

df.to_csv(FINAL_SAVE_PATH, index=False, encoding='utf-8')



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
regression_summary_df.to_csv("output_/output_dpsk/no_control_dpsk-regression.csv", index=False)

