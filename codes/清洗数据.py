import json

# 读取两个JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1909_1911-1916_cleaned.json', 'r') as f1:
    data1 = json.load(f1)


all_results = []

for person in data1:
    if "quote" in person.keys():
        person["comment"] = person["comment"]+person["quote"]
    if "comments" in person.keys():
        person["comment"] = person["comment"]
        del person["comments"]
    if not person["comment"] == "":
        all_results.append(person)




with open('1909_1911-1916_clean.json', 'w') as out_file:
    json.dump(all_results, out_file, ensure_ascii=False, indent=2)