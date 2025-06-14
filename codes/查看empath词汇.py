from empath import Empath
import json

# 读取自定义维度和新词
with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

# 提取所有自定义类别
CUSTOM_CATEGORIES = []
for key, value in framework.items():
    CUSTOM_CATEGORIES.extend(value)

# 创建 Empath 对象，并添加自定义类别
MODEL = "fiction"
lexicon = Empath()
for category in CUSTOM_CATEGORIES:
    if category not in lexicon.cats:
        lexicon.create_category(category, NEW_WORDS[category], model=MODEL)

# 构建输出字典
category_words = {cat: list(lexicon.cats.get(cat, [])) for cat in CUSTOM_CATEGORIES}
output_data = {
    "categories": CUSTOM_CATEGORIES,
    "category_words": category_words
}

# 保存为 JSON 文件
with open("empath_output.json", "w", encoding="utf-8") as f_out:
    json.dump(output_data, f_out, ensure_ascii=False, indent=2)

print("Saved to empath_output.json.")
