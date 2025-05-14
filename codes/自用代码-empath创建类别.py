from empath import Empath
lexicon = Empath()
l1 = lexicon.create_category("appearance",["beautiful","winsome","blonde","handsome","pretty","cute","gorgeous","elegant","charming","lovely","attractive"], model = "nytimes")
l2 = lexicon.create_category("appearance",["beautiful","winsome","blonde","handsome","pretty","cute","gorgeous","elegant","charming","lovely","attractive"], model = "fiction")
l3 = lexicon.create_category("appearance",["beautiful","winsome","blonde","handsome","pretty","cute","gorgeous","elegant","charming","lovely","attractive"], model = "reddit")
for word in l1:
    if word in l2:
        if word in l3:
            l1.remove(word)
            l2.remove(word)
            l3.remove(word)
print("l1:",l1)
print("l2:",l2)
print("l3:",l3)
            

