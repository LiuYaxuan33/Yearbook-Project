import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

def custom_color_func_blue(word, font_size, position, orientation, random_state=None, **kwargs):
    # H, S 随机（或固定），L 在 40%–70% 之间
    h = random.randint(200, 240)      # 取一个蓝色系的色相
    s = random.randint(60, 90)       # 饱和度 70%–100%
    l = random.randint(40, 70)        # 亮度 40%–70%
    return f"hsl({h}, {s}%, {l}%)"
def custom_color_func_red(word, font_size, position, orientation, random_state=None, **kwargs):
    # H, S 随机（或固定），L 在 40%–70% 之间
    h = random.randint(0, 20)      # 取一个红色系的色相
    s = random.randint(70, 100)       # 饱和度 70%–100%
    l = random.randint(40, 70)        # 亮度 40%–70%
    return f"hsl({h}, {s}%, {l}%)"




# 1. 读取数据
df = pd.read_csv('output_/output_basic/1-grams.csv')

# 2. 根据系数正负分离词语
pos_df = df[df['coefficient'] > 0]
neg_df = df[df['coefficient'] < 0]

# 3. 构造词频字典（使用绝对系数值）
pos_freq = dict(zip(pos_df['word'], pos_df['coefficient']))
neg_freq = dict(zip(neg_df['word'], -(neg_df['coefficient'])))

# 4. 配置并生成词云
wc_pos = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    max_words=200,
    contour_width=3,
    contour_color='steelblue',
    scale=2,
    collocations=False,
    color_func = custom_color_func_red
).generate_from_frequencies(pos_freq)

wc_neg = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    max_words=200,
    contour_width=3,
    contour_color='firebrick',
    scale=2,
    collocations=False,
    color_func = custom_color_func_blue
).generate_from_frequencies(neg_freq)

# 5. 绘制
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor='#f0f0f0')
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title('Female', fontsize=20, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title('Male', fontsize=20, fontweight='bold')
axes[1].axis('off')

plt.tight_layout(pad=5)
plt.show()
