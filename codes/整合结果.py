import json

# 读取两个JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1911.json', 'r') as f1:
    data1 = json.load(f1)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1912.json', 'r') as f2:
    data2 = json.load(f2)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1913.json', 'r') as f3:
    data3 = json.load(f3)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1914.json', 'r') as f4:
    data4 = json.load(f4)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1915.json', 'r') as f5:
    data5 = json.load(f5)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1916.json', 'r') as f6:
    data6 = json.load(f6)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1909.json', 'r') as f7:
    data7 = json.load(f7)

with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/final_output/1906.json', 'r') as f8:
    data8 = json.load(f8)

all_results = []

for person in data1:
    person["year"] = 1911
    all_results.append(person)

for person in data2:
    person["year"] = 1912
    all_results.append(person)

for person in data3:
    person["year"] = 1913
    all_results.append(person)

for person in data4:
    person["year"] = 1914
    all_results.append(person)

for person in data5:
    person["year"] = 1915
    all_results.append(person)

for person in data6:
    person["year"] = 1916
    all_results.append(person)

for person in data7:
    person["year"] = 1909
    all_results.append(person)

for person in data8:
    person["year"] = 1906
    all_results.append(person)

with open('1906_1909_1911-1916.json', 'w') as out_file:
    json.dump(all_results, out_file, ensure_ascii=False, indent=2)