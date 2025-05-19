import json

with open('all_data_use.json', 'r', encoding='utf-8') as f:
    data = json.load(f)



a = 0
b = 0
# 处理每个人员信息
for i, person in enumerate(data):
    if person['major'] == "Economics":
        if person['gender'] == "Male":
            a += 1
        else:   
            b += 1
print("经济学专业男生人数：", a)
print("经济学专业女生人数：", b)
