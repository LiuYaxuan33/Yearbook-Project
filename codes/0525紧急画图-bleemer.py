import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


output_df = pd.read_csv("output_/output_bleemer/genderness_in_majors-median.csv")

print(output_df[output_df['standardized_pred'] == output_df['standardized_pred'].min()])

# 1. 统计：按专业和性别分组，计算均值和标准误
grouped_stats = (
    output_df
    .groupby(['major', 'gender'])['standardized_pred']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
grouped_stats['se'] = grouped_stats['std'] / np.sqrt(grouped_stats['count'])


# 2. 图形绘制：条形图，一上一下表示不同性别

# 为绘图做准备：变成专业 × 性别的多层结构
pivot_df = grouped_stats.pivot(index='major', columns='gender', values='mean')
pivot_se = grouped_stats.pivot(index='major', columns='gender', values='se')
# 获取样本数，用于图中标注
pivot_count = grouped_stats.pivot(index='major', columns='gender', values='count')
pivot_count = pivot_count.fillna(0)

# 专业排序（仍然按女性平均值）
sorted_majors = pivot_df['Female'].sort_values().index.tolist()

x = np.arange(len(sorted_majors))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# 女生柱子（向上）
female_vals = pivot_df.loc[sorted_majors, 'Female']
female_se = pivot_se.loc[sorted_majors, 'Female']
female_n = pivot_count.loc[sorted_majors, 'Female']

bars1 = ax.bar(x, female_vals, width=width, color='lightcoral', label='Female',
               yerr=female_se, capsize=5)

# 男生柱子（向下）
male_vals = pivot_df.loc[sorted_majors, 'Male']
male_se = pivot_se.loc[sorted_majors, 'Male']
male_n = pivot_count.loc[sorted_majors, 'Male']

bars2 = ax.bar(x+0.2, -male_vals, width=width, color='skyblue', label='Male',
               yerr=male_se, capsize=5)



# 图表样式
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sorted_majors, rotation=45, ha='right')
ax.set_ylabel("Standardized Prediction (z-score)")
ax.set_title("Standardized Lasso Gender Prediction by Major and Gender")
ax.legend()
plt.tight_layout()
plt.savefig("output_/output_bleemer/genderness in majors-median.png")
plt.show()
output_df.to_csv("output_/output_bleemer/genderness_in_majors-median.csv", index=False)