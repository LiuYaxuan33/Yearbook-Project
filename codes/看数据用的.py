import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取文件
file_path = "output_/output_dpsk/deepseek_final.csv"
df = pd.read_csv(file_path)

# 2. 提取最后五个以 "mean" 结尾的类别
mean_columns = [
    'Ability_and_Competence_mean',
    'Character_or_Morality_mean',
    'Appearance_mean',
    'Social_Relations_mean',
    'Activities_Engagement_mean'
]

# 3. 按性别分组，计算每个类别的均值
gender_means = df.groupby('gender')[mean_columns].mean()

# 4. 准备雷达图的标签和角度
labels = mean_columns
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合多边形

# 5. 创建雷达图
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# 6. 绘制每个性别的曲线和填充
for gender in gender_means.index:
    values = gender_means.loc[gender].tolist()
    values += values[:1]  # 闭合多边形
    ax.plot(angles, values, label=f'Gender {gender}')
    ax.fill(angles, values, alpha=0.1)

# 7. 设置坐标轴标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels([
    'Ability_and_Competence',
    'Character_or_Morality',
    'Appearance',
    'Social_Relations',
    'Activities_Engagement'
], fontsize=10)
ax.set_title('Gender Difference on Parent Categories', size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.show()
