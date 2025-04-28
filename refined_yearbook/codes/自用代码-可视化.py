import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/results/2-5.csv')
# 拆分正负系数
pos = df[df['coefficient'] >= 0].sort_values('coefficient', ascending=True)
neg = df[df['coefficient'] < 0].sort_values('coefficient', ascending=True)

# 设置动态画布高度
bar_height = 0.4  # 每条数据高度增加
min_height = 6
max_rows = max(len(pos), len(neg))
fig_height = max(min_height, bar_height * max_rows)

# 创建子图：更宽、更舒展
fig, axes = plt.subplots(ncols=2, sharey=False, figsize=(25, fig_height))

# 左侧正系数
axes[0].barh(pos['word'], pos['coefficient'], color='indianred')
axes[0].invert_yaxis()
axes[0].set_title('Positive Coefficients', fontsize=14)
axes[0].set_xlabel('Coefficient', fontsize=12)
axes[0].tick_params(labelsize=24)

# 右侧负系数
axes[1].barh(neg['word'], neg['coefficient'], color='steelblue')
axes[1].invert_yaxis()
axes[1].set_title('Negative Coefficients', fontsize=14)
axes[1].set_xlabel('Coefficient', fontsize=12)
axes[1].tick_params(labelsize=20)

# 整体布局美化
save_path = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/lasso_coeff_plot-2.jpg'

plt.tight_layout(pad=3.0)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

