import json
import os
from dotenv import load_dotenv
import pandas as pd

# 加载环境变量
load_dotenv()

# 从字符串加载 JSON 字典
MAJOR_CATEGORIES = json.loads(os.getenv("MAJOR_CATEGORIES"))

# 读取数据
with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 添加 is_{category} 列
for category, majors in MAJOR_CATEGORIES.items():
    df[f'is_{category}'] = df['major'].isin(majors).astype(int)

# 保留至少命中一个分类的行
is_cols = [f'is_{category}' for category in MAJOR_CATEGORIES]
df = df[df[is_cols].sum(axis=1) > 0].copy()

base_fields = ['name', 'gender', 'major', 'hometown', 'clubs', 'comment', 'year']

label_fields = [f'is_{category}' for category in MAJOR_CATEGORIES]

df = df[base_fields + label_fields]


# 保存为 JSON
with open('all_data_use_labeled.json', 'w', encoding='utf-8') as out_file:
    json.dump(df.to_dict(orient='records'), out_file, ensure_ascii=False, indent=2)
