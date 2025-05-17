import json

with open('all_data_use.json', 'r', encoding='utf-8') as f1:
    data2 = json.load(f1)

all_clubs = []
all_majors = []
all_hometowns = []
all_genders = []
# 处理每个人员信息
for i, person in enumerate(data2):
    for club in person['clubs']:
        if club not in all_clubs:
            all_clubs.append(club)
    if person['hometown'] not in all_hometowns:
        all_hometowns.append(person['hometown'])
    if person['major'] not in all_majors:
        all_majors.append(person['major'])
    if person['gender'] not in all_genders:
        all_genders.append(person['gender'])
    #if not person['comment'] == "":
    #print(i)
all_clubs = sorted(all_clubs)
all_majors = sorted(all_majors)
all_hometowns = sorted(all_hometowns)
datas = [all_clubs, all_hometowns, all_majors, all_genders]
with open('data_profile.json', 'w', encoding='utf-8') as out_file:
    json.dump(datas, out_file, ensure_ascii=False, indent=2)