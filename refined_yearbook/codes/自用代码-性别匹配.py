import json

# 读取两个JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1913_new.json', 'r') as f1:
    data1 = json.load(f1)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1913.json', 'r') as f2:
    data2 = json.load(f2)

# 处理每个人员信息
for i, person in enumerate(data2):
    # 计算对应的图片索引和性别索引
    pic_index = i // 4
    gender_index = i % 4
    
    # 获取图片性别信息
    pic_gender = data1[pic_index]['genders'][gender_index]
    
    # 重命名原gender字段
    person['name_gender'] = person.pop('gender')
    
    # 添加新字段
    person['pic_gender'] = pic_gender
    person['gender'] = person['name_gender'] if person['name_gender'] == pic_gender else 'BUZHIDAO'

# 保存结果到新文件
with open('output.json', 'w') as out_file:
    json.dump(data2, out_file, ensure_ascii=False, indent=2)