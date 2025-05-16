import json

with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)



N = 0
# 处理每个人员信息
for i, person in enumerate(data):
    if "Iowa" in person['hometown']:
        N += 1
        print(f"Name: {person['name']}")
print(f"Total number of people from Iowa: {N}")
