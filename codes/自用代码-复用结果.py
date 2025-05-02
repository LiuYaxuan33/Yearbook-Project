import json

# 读取两个JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1913_new.json', 'r') as f1:
    data1 = json.load(f1)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1913_.json', 'r') as f2:
    data2 = json.load(f2)

# 处理每个人员信息
for i, person in enumerate(data2):
    person["gender"] = data1[i]["gender"]
    person["name_gender"] = data1[i]["name_gender"]
    person["pic_gender"] = data1[i]["pic_gender"]

# 保存结果到新文件
with open('1913.json', 'w') as out_file:
    json.dump(data2, out_file, ensure_ascii=False, indent=2)