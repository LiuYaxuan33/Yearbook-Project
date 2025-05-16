# Yearbook-Project

## 250516

### sentiment dimensions

Based on Trix & Psenka (2003) and Schmader et al. (2007) and LIWC (2001)

EJ_REFERENCE_CATEGORIES = [
    "ability","grindstone","research","standout","teaching&Citizenship","Recuitment Prospects"
]

CUSTOM_CATEGORIES = [
    "ability","grindstone","school","standout","citizenship","positive_emotion","friends","trust","appearance"
]

Empath has: "school", "positive_emotion", "friends", "trust"

Customly created categories and root words:
    "ability": 
        ["ability", "briliant", "intelligent", "talented", "gifted", "creative"],
    "grindstone": 
        ["grindstone","hardworking", "diligent", "conscientious", "persevering", "persistent", "determined"],
    "standout": 
        ["standout", "excellent", "outstanding", "superb", "exceptional", "top", "rising_star", "among_the_best", "truly_exceptional"],
    "citizenship": 
        ["citizenship", "community", "society", "supportive", "strong communicator", "dedicated", "a_pleasure_to_work_with"],
    "appearance": 
        ["appearance", "looks", "physical_appearance", "beauty", "handsome", "pretty","beautiful", "attractive", "gorgeous", "cute","winsome", "eye", "eyes", "cheek", "cheeks", "nose", "lips"]
    
    the first four categories' root words come from Eberhardt et al.(2023), unfortunately they didn't give out complete lexicon in their paper. for "appearnce", chatGPT gave the rootwords above.

    word embedding model: fiction (reddit? nytimes? seemingly similar. as robustness check?)

    adding or deleting one category won't affect analysis on other categories.

### new 2-grams: 

1. higher min-DF: not effective, "man_big" appears for more than 10 times
2. no stop-words: not effective, lots of meaningless results
3. new logic: consider stop-words, but only include 2-grams whose components are also adjacent in the original text
    - example: "The extremely talented student gave a brilliant presentation."
    - stop-words here includes: the, a
    - old logic generates 2-grams(with adj, adv and none):
        - extremely_talented  
        - talented_student  
        - student_gave  
        - gave_brilliant  
        - brilliant_presentation: not good!
    - new logic gives:
        - extremely_talented  
        - talented_student  
        - student_gave
        - brilliant_presentation
    - the result seems quite reasonable

\"I like above all things in the world to be loved.\" This fair maiden hails from Gotch’s town. Has many strings to her beau. Sometimes keeps her Romeo in suspense. However, only true worth attracts admiration.

### llm consistency: 10*10 repeat experiment

    consistency problem is more about person/comment, not categories

### majors:
    [
    "",(yearbook didn't give his major)
    "Agricultural Education",
    "Agricultural Engineering",
    "Agronomy",
    "Animal Husbandry",
    "Animal Husbandry and Agricultural Education",
    "Ceramics",
    "Chemistry",
    "Chemistry Engineering",
    "Civil Engineering",
    "Colonials",
    "Dairy",
    "Dairy Husbandry",
    "Dairy Husbandry-Animal Husbandry",
    "Domestic Science",
    "Economics",
    "Electrical Engineering",
    "Forestry",
    "General Science",
    "General and Domestic Science",
    "Home Economics",
    "Horticulture",
    "Horticulture and Forestry",
    "Industrial Chemistry",
    "Industrial Science",
    "Mechanical & Electrical Engineering",
    "Mechanical Engineering",
    "Mining Engineering",
    "Music",
    "Science",
    "Science and Agronomy",
    "Special",
    "Str. Des",(not sure what this means, "structural design"?)
    "Veterinary",
    "Veterinary Medicine",
    "Veterinary and Animal Husbandry"
  ],

how to devide? chatGPT gives STEM majors:
    Chemistry
    General Science
    Industrial Chemistry
    Industrial Science
    Science
    Science and Agronomy
    Agricultural Engineering
    Ceramic(s)
    Chemical Engineering / Chemistry Engineering
    Civil Engineering
    Electrical Engineering
    Mechanical & Electrical Engineering
    Mechanical Engineering
    Mining Engineering
    Str. Des

should agricultural majors be counted as STEM?
    Agricultural Education
    Agronomy
    Animal Husbandry
    Dairy
    Dairy Husbandry
    Dairy Husbandry-Animal Husbandry
    Horticulture
    Horticulture and Forestry
    Forestry
and Veterinary:
    Veterinary
    Veterinary Medicine
    Veterinary and Animal Husbandry

### about regression:

    1. lasso with control variables?
    2. (logical) dependant variables: sentiment score or words?
    3. factors other than gender: major, hometown, and year
    4. technical detail: linear or logistic?

