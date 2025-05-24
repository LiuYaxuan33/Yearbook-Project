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
print(df.keys())
df = df.drop(columns=['name', 'year','gender','is_agriculture', 'is_home_economics',
       'is_science', 'is_engineering', 'is_music', 'is_education',
       'is_veterinary', 'hometown_Iowa'])

df = (df-df.min())/(df.max()-df.min())  # 标准化处理

df.to_csv('output_/output_agreement_analysis.csv', index=False, encoding='utf-8-sig')

with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

dims = []

for key, value in framework.items():
    dims.extend(value)
    dims.append(key + "_mean")  # 添加大类别均值



# 4) 保存结果的列表
results = []

# 5) 显著性标记函数
def significance_marker(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

# 6) 计算各维度指标
for d in dims:
    col1 = f'{d}_m1'
    col2 = f'{d}_m2'
    sub = df[[col1, col2]].dropna()
    x = sub[col1]
    y = sub[col2]
    
    # 计算相关性和差异
    r, p_r = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)
    t_stat, p_t = ttest_rel(x, y)
    mean_diff = np.mean(x - y)
    
    # 添加显著性标记
    pearson_star = significance_marker(p_r)
    spearman_star = significance_marker(p_s)
    
    results.append({
        'Dimension': d,
        'Pearson_r': f"{r:.3f} {pearson_star}",
        'Pearson_p': f"{p_r:.3f}",
        'Spearman_rho': f"{rho:.3f} {spearman_star}",
        'Spearman_p': f"{p_s:.3f}",
        'T_stat': f"{t_stat:.3f}",
        'T_p': f"{p_t:.3f}",
        'Mean_diff': f"{mean_diff:.3f}"
    })

# 7) 转为 DataFrame
results_df = pd.DataFrame(results)

# 8) 生成 LaTeX 表格代码
latex_table = results_df.rename(columns={
    'Dimension': '维度',
    'Pearson_r': 'Pearson $r$',
    'Pearson_p': '$p_r$',
    'Spearman_rho': 'Spearman $\\rho$',
    'Spearman_p': '$p_\\rho$',
    'T_stat': '$t$',
    'T_p': '$p_t$',
    'Mean_diff': '平均差值'
}).to_latex(index=False, float_format="%.3f", caption='两种口径评分一致性分析', label='tab:agreement_analysis')

print(latex_table)