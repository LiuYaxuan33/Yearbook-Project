import json

# 读取两个JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/all_data_use.json', 'r') as f1:
    data1 = json.load(f1)

a = 0
# 处理每个人员信息
for i, person in enumerate(data1):
    if "loyal" in person['comment'] and "student" in person['comment']:
        a = a + 1
        print(a)
        print(person['gender'])
        print(person['comment'])



'''
1
Male
He has as yet started no great conflagrations. His sweet, confiding smile wins acquaintances, and he does nothing to scare them away. An able student, loyal to his society and to I. S. C.
2
Male
"Wroe" is a hand-me-down from the '11's, and one whom we are heartily glad to welcome to our ranks. There's a world of fun in those quiet eyes; and he'll blush like a co-ed when girls are mentioned. He's a model student, a loyal Ag., and a fellow whom it is a pleasure to know.
3
Male
"War is a terrible trade, but when on dress parade, sweet is the smell of the powder." "Red" entered I. S. C. with a longing for military glory, and we think his ambition was realized when he wore his uniform home during Christmas vacation. Has the reputation of having been stretched more than any other man in school. A good student and a loyal Ag.
4
Male
"Good things are put up in small parcels." This little man has a habit of getting everything as he goes. He is as wise as an owl but does not "Hoot" like one; far from it. When he has something to say, it is worth hearing. A good student and a loyal Engineer.
5
Male
"Enlarge him and make a friend of him" Kirk was run in the common mold. Nothing extraordinary about him. Just a good friend, a faithful student, and a loyal supporter of Ames.
'''