import json

with open('all_data_use_labeled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


ALL_LABELS = [
    "is_agriculture",
    "is_home economics",
    "is_science",
    "is_engineering",
    "is_music",
    "is_education",
    "is_veterinary"]

MAJOR_TO_ANALYZE = ALL_LABELS[0] 

for labels in ALL_LABELS:
    man = 0
    woman = 0
    for person in data:
        if person[labels] == 1:
            if person['gender'] == 'Male':      
                man += 1
            else:  
                woman += 1
    print(f"{labels}: man has {man} and woman has {woman}")

# 统计结果可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 构建统计数据列表
stats = []

for labels in ALL_LABELS:
    man = 0
    woman = 0
    for person in data:
        if person[labels] == 1:
            if person['gender'] == 'Male':
                man += 1
            else:
                woman += 1
    stats.append({'major': labels.replace("is_", "").replace("_", " ").title(), 'gender': 'Male', 'count': man})
    stats.append({'major': labels.replace("is_", "").replace("_", " ").title(), 'gender': 'Female', 'count': woman})

# 转换为 DataFrame
df = pd.DataFrame(stats)

# 设置画图风格
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))

# 使用 seaborn 画分组条形图
barplot = sns.barplot(
    data=df,
    x='major',
    y='count',
    hue='gender',
    palette='Set2'
)

# 添加数值标签
for container in barplot.containers:
    barplot.bar_label(container, fmt='%d', label_type='edge', padding=3)

# 图表美化
plt.title('Gender Distribution across Majors')
plt.xlabel('Major')
plt.ylabel('Number of Students')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.legend(title='Gender')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# 展示图形
plt.show()
plt.savefig('output_/output_basic/descriptive_statistics.png', dpi=300, bbox_inches='tight')
