import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_rel
import matplotlib.pyplot as plt
import json

# 1) 读取 CSV
df1 = pd.read_csv('output_/output_dpsk/deepseek_final.csv')   # 第一种口径
df2 = pd.read_csv('output_/output_sentiment_tfidf/tfidf.csv')   # 第二种口径

# 2) 合并
df = pd.merge(df1, df2, on=['name', 'gender'], suffixes=('_m1', '_m2'))

# 3) 读取维度信息
with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

dims = []
for key, value in framework.items():
    dims.extend(value)
    dims.append(key + "_mean")

# 初始化最大差异记录
max_abs_diff = -1
max_info = {}

# 遍历所有维度和样本，找出绝对差值最大的项
for d in dims:
    col1 = f'{d}_m1'
    col2 = f'{d}_m2'
    
    if col1 in df.columns and col2 in df.columns:
        sub = df[['name', 'gender', col1, col2]].dropna()
        diffs = (sub[col1] - sub[col2]).abs()
        
        if not diffs.empty:
            max_idx = diffs.idxmax()
            diff_val = diffs.loc[max_idx]
            
            if diff_val > max_abs_diff:
                max_abs_diff = diff_val
                max_info = {
                    'name': sub.loc[max_idx, 'name'],
                    'gender': sub.loc[max_idx, 'gender'],
                    'dimension': d,
                    'score_m1': sub.loc[max_idx, col1],
                    'score_m2': sub.loc[max_idx, col2],
                    'diff': sub.loc[max_idx, col1] - sub.loc[max_idx, col2]
                }

# 输出结果
print("差异最大的单个元素如下：")
print(f"姓名：{max_info['name']}")
print(f"性别：{max_info['gender']}")
print(f"维度：{max_info['dimension']}")
print(f"口径1得分：{max_info['score_m1']:.4f}")
print(f"口径2得分：{max_info['score_m2']:.4f}")
print(f"差值（m1 - m2）：{max_info['diff']:.4f}")
print(f"绝对差值：{max_abs_diff:.4f}")
