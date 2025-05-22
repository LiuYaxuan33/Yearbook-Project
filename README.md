# Yearbook-Project

## To those who are interested:

This project is based on yearbook of year 1906, 1909 and 1911-1916 of Iowa State's University. We hope to, using some NLP algorithms, find language of stereotype used in graduates' personal comments. Such comments are typically made by a committe of students, thus are made from peer's perspective. The method we are applying include LASSO regression, tf-idf-based and llm-based sentiment analysis (with lexicon Empath and llm Deepseek-v3)

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

### llm consistency: 10\*10 repeat experiment

    consistency problem is more about person/comment, not categories

### majors:

agriculture
home economics
science
engineering
music
education
veterinary
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
    5. what can we do with lasso-selected features? classify with categories above?

### partial-lasso

majors are too strong as controls, so the residuals left for word selection are small, given that control variables are not panelized, regression coefficients will be focused on controls

hometown as control: how to cluster? about 1850/2120 samples are from Iowa.

is partial-lasso effective? this may destroy some fine properties of lasso loss function.

can we just add control, and panelize them as what we did to word features?

### about feature selection: SVM (support vector machine)

explanable: lagrange multiplier

s~l~o~w~s~o~s~l~o~w~

mathematically more elegant, and effective

## 250519

### about dimensions

CUSTOM_CATEGORIES = [
"ability","grindstone","school","standout","citizenship","positive_emotion","friends","trust","appearance"
]
optimism
sports

here I give out the label structured dimensions of comment sentiment:

OVERALL_EMOTIONS
positive_emotion
CHARACTER & MORALITY
trust
citizenship
optimism
ABILITY & PERSONAL TRAITS
ability
grindstone
standout
SOCIAL RELATIONSHIPS
friends
trust
PARTICIPATION & ENGAGEMENT
school
sports
APPEARANCE
subjective_appearance
physical_features

**ATTENTION: MORE LITERATURE WORK IS NEEDED!**

### llm consistency

larger sample
you'll need to list some examples, as testimony to the limitation of usage of llm in your research

### majors

agriculture
"Agronomy",
"Animal Husbandry",
"Animal Husbandry and Agricultural Education",
"Dairy",
"Dairy Husbandry",
"Dairy Husbandry-Animal Husbandry",
"Science and Agronomy",
"Veterinary and Animal Husbandry"
"Forestry",
"Horticulture",
"Horticulture and Forestry",
home economics
"Domestic Science",
"General and Domestic Science",
"Home Economics",
"Economics",
science
"Chemistry",
"Science",
"Science and Agronomy",
"General Science",
"General and Domestic Science",
"Industrial Chemistry",
"Industrial Science",
engineering
"Chemistry Engineering",
"Civil Engineering",
"Agricultural Engineering",
"Ceramics",
"Electrical Engineering",
"Mechanical & Electrical Engineering",
"Mechanical Engineering",
"Mining Engineering",
music
"Music",
education
"Animal Husbandry and Agricultural Education",
"Agricultural Education",
veterinary
"Veterinary",
"Veterinary Medicine",
"Veterinary and Animal Husbandry"

### lasso

regression by group

### other

do clubs matter? specially, does whether one student is among the bomb committe matter?

it seems meaningless to consider hometown...

do year of graduation matter? the Great War?

## 250520

### regression by group

seems impossible:
agriculture: man has 705 and woman has 6
home economics: man has 3 and woman has 284
science: man has 41 and woman has 78
engineering: man has 834 and woman has 2
music: man has 0 and woman has 9
education: man has 25 and woman has 1
veterinary: man has 142 and woman has 1

either extremely imbalance or too few samples

science major can generate (to the least extent) reasonable results, but doesn't seem meaningful...

can we do case analysis?

### stereotype of majors

something puzzling seems to go wrong when running on agriculture major...there are no feature selected...i dont know why

I KNOW! sometimes the selected alpha goes larger than 0.001, and that panalizes coefficients too much.

### sentiment dimensions

DeepResearch give a sentiment dimension system:

能力 (Ability/Competence)

智力 (Intelligence) (理论来源：大五人格–开放性/智力因素)
：intelligent, clever, bright, smart, astute, perceptive, insightful, sharp, brainy, knowledgeable

勤奋 (Diligence) (理论来源：大五人格–尽责性)
：diligent, hardworking, industrious, meticulous, thorough, persistent, dedicated, organized, disciplined, responsible

创造力 (Creativity) (理论来源：大五人格–开放性)
：creative, imaginative, innovative, original, artistic, resourceful, inventive, ingenious, visionary, inspired

领导力 (Leadership/Agency) (理论来源：大五人格–外向性；刻板印象模型–能力维度)
：leader, assertive, confident, decisive, ambitious, charismatic, inspiring, authoritative, strategic, commanding

品格 (Character/Morality)

诚实 (Honesty/Integrity) (理论来源：HEXACO 模型–诚实-谦逊维度)
：honest, truthful, sincere, trustworthy, genuine, ethical, principled, honorable, upright, fair-minded

善良 (Kindness/Compassion) (理论来源：道德基础理论–关怀/损害)
：kind, caring, compassionate, generous, empathetic, considerate, gentle, supportive, charitable, warm-hearted

忠诚 (Loyalty/Dependability) (理论来源：道德基础理论–忠诚/背叛)
：loyal, faithful, devoted, steadfast, committed, reliable, supportive, dedicated, unwavering, constant

公平 (Fairness/Justice) (理论来源：道德基础理论–公平/欺骗)
：fair, just, impartial, equitable, unbiased, objective, righteous, principled, moderate, balanced

外貌 (Appearance)

吸引力 (Attractiveness) (理论来源：社会心理学–光环效应)
：handsome, beautiful, attractive, pretty, charming, elegant, graceful, lovely, radiant, gorgeous

整洁 (Neatness/Grooming) (理论来源：身体形象研究)
：neat, tidy, well-groomed, clean, polished, well-dressed, organized, presentable, trim, spotless

体格 (Physique/Strength) (理论来源：身体形象研究)
：strong, athletic, fit, robust, sturdy, healthy, muscular, vigorous, energetic, resilient

社交关系 (Social Relations/Warmth)

友好 (Friendliness) (理论来源：大五人格–宜人性；刻板印象模型–温暖维度)
：friendly, amiable, warm, approachable, congenial, pleasant, affable, genial, kindhearted, sociable

热情 (Energetic/Outgoing) (理论来源：大五人格–外向性)
：outgoing, energetic, lively, enthusiastic, animated, vibrant, vivacious, spirited, bubbly, dynamic

合作 (Cooperativeness) (理论来源：大五人格–宜人性)
：cooperative, helpful, supportive, collaborative, considerate, accommodating, flexible, team-oriented, empathetic, understanding

幽默 (Humor) (理论来源：社会心理学–社交沟通)
：humorous, witty, funny, jovial, entertaining, amusing, lighthearted, playful, comical, cheerful

活动参与 (Activities/Engagement)

运动 (Athletic Participation) (理论来源：社会心理学–体能参与)
：athletic, sporty, agile, fit, strong, robust, vigorous, active, nimble, competitive

文艺 (Arts/Cultural Activities) (理论来源：社会心理学–艺术参与)
：artistic, musical, creative, talented, expressive, cultured, aesthetic, inventive, imaginative, visionary

学术 (Academic Involvement) (理论来源：教育心理学–学术成就)
：studious, scholarly, intellectual, educated, knowledgeable, analytical, learned, erudite, literate, bookish

社团 (Club/Service Participation) (理论来源：社会心理学–社区参与)
：involved, engaged, active, committed, dedicated, enthusiastic, volunteer, service-oriented, civic-minded, helpful

Ability/Competence

1. Intelligence

   - intelligent, clever, bright, smart, astute, perceptive, insightful, sharp, brainy, knowledgeable

2. Grindstone

   - diligent, hardworking, industrious, meticulous, thorough, persistent, dedicated, organized, disciplined, responsible, conscientious, persevering, determined

3. Creativity

   - creative, imaginative, innovative, original, artistic, resourceful, inventive, ingenious, visionary, inspired

4. Leadership/Agency
   - leader, assertive, confident, decisive, ambitious, charismatic, inspiring, authoritative, strategic, commanding

Character/Morality

1. Honesty/Integrity

   - honest, truthful, sincere, trustworthy, genuine, ethical, principled, honorable, upright, fair-minded

2. Kindness/Compassion

   - kind, caring, compassionate, generous, empathetic, considerate, gentle, supportive, charitable, warm-hearted

3. Loyalty/Dependability

   - loyal, faithful, devoted, steadfast, committed, reliable, supportive, dedicated, unwavering, constant

4. Fairness/Justice
   - fair, just, impartial, equitable, unbiased, objective, righteous, principled, moderate, balanced

Appearance

1. Attractiveness

   - handsome, beautiful, attractive, pretty, charming, elegant, graceful, lovely, radiant, gorgeous

2. Neatness

   - neat, tidy, well-groomed, clean, polished, well-dressed, organized, presentable, trim, spotless

3. Strength

   - strong, athletic, fit, robust, sturdy, healthy, muscular, vigorous, energetic, resilient, big, hard

4. Physical Features
   - looks, physical_appearance, eye, cheek, nose, lips, hair, blue, brown, black, blonde

Social Relations/Warmth

1. Friendliness

   - friendly, amiable, warm, approachable, congenial, pleasant, affable, genial, kindhearted, sociable

2. Energetic/Outgoing

   - outgoing, energetic, lively, enthusiastic, animated, vibrant, vivacious, spirited, bubbly, dynamic

3. Cooperativeness

   - cooperative, helpful, supportive, collaborative, considerate, accommodating, flexible, team-oriented, empathetic, understanding

4. Humor
   - humorous, witty, funny, jovial, entertaining, amusing, lighthearted, playful, comical, cheerful, laugh, giggle, joke

Activities/Engagement

1. Athletic Participation

   - athletic, sporty, agile, fit, strong, robust, vigorous, active, nimble, competitive

2. Arts/Cultural Activities

   - artistic, musical, creative, talented, expressive, cultured, aesthetic, inventive, imaginative, visionary

3. Academic Involvement

   - studious, scholarly, intellectual, educated, knowledgeable, analytical, learned, erudite, literate, bookish

4. Club/Service Participation
   - involved, engaged, active, committed, dedicated, enthusiastic, volunteer, service-oriented, civic-minded, helpful

### FINAL_VERSION of sentiment dimension

{
   "Ability & Competence":
      ["intelligence", "grindstone", "creativity", "leadership"],
   "Character/Morality":
      ["honesty", "kindness", "dependability", "justice"],
   "Appearance":
      ["attractiveness", "neatness", "strength", "physical features"],
   "Social Relations":
      ["friends", "outgoing", "cooperativeness", "humor"],
   "Activities/Engagement":
      ["sports", "art and culture", "school", "clubs"]
}



Ability/Competence

Intelligence

intelligent, clever, bright, smart, astute, perceptive, insightful, sharp, brainy, knowledgeable, intelligence, insight, knowledge, wisdom, acumen, aptitude, savvy, proficiency, erudition, cognition

Grindstone

diligent, hardworking, industrious, meticulous, thorough, persistent, dedicated, organized, disciplined, responsible, diligence, perseverance, discipline, effort, dedication, industriousness, work ethic, stamina, tenacity, rigor, conscientious, persevering, determined

Creativity

creative, imaginative, innovative, original, artistic, resourceful, inventive, ingenious, visionary, inspired, creativity, imagination, innovation, originality, artistry, ingenuity, inspiration, vision, resourcefulness, flair

Leadership

leader, assertive, confident, decisive, ambitious, charismatic, inspiring, authoritative, strategic, commanding, leadership, authority, initiative, ambition, charisma, decisiveness, strategy, drive, governance, stewardship

Character/Morality

Honesty/Integrity

honest, truthful, sincere, trustworthy, genuine, ethical, principled, honorable, upright, fair-minded, honesty, integrity, sincerity, trustworthiness, ethics, honor, principle, transparency, virtue, fairness

Kindness/Compassion

kind, caring, compassionate, generous, empathetic, considerate, gentle, supportive, charitable, warm-hearted, kindness, compassion, empathy, generosity, mercy, charity, benevolence, sympathy, goodwill, understanding

Loyalty/Dependability

loyal, faithful, devoted, steadfast, committed, reliable, supportive, dedicated, unwavering, constant, loyalty, fidelity, devotion, reliability, commitment, steadfastness, faithfulness, trust, support, allegiance

Fairness/Justice

fair, just, impartial, equitable, unbiased, objective, righteous, principled, moderate, balanced, justice, fairness, impartiality, equity, objectivity, righteousness, balance, moderation, rule of law, judiciary

Appearance

Attractiveness

handsome, beautiful, attractive, pretty, charming, elegant, graceful, lovely, radiant, gorgeous, beauty, charm, elegance, grace, allure, glamour, radiance, appeal, poise, presence

Neatness

neat, tidy, well-groomed, clean, polished, well-dressed, organized, presentable, trim, spotless, neatness, tidiness, grooming, polish, orderliness, cleanliness, presentation, refinement, sharpness, crispness

Strength

strong, athletic, fit, robust, sturdy, healthy, muscular, vigorous, energetic, resilient, big, hard, strength, fitness, vigor, stamina, robustness, health, resilience, power, endurance, musculature

Physical Features

physical_appearance, looks, visage, features, complexion, hair, eyes, eyebrows, cheekbones, jawline, posture, physique, eye, cheek, nose, lips, hair, blue, brown, black, blonde

Social Relations/Warmth

Friendliness->friends

friendly, amiable, warm, approachable, congenial, pleasant, affable, genial, kindhearted, sociable, friendliness, amity, warmth, approachability, geniality, cordiality, hospitality, camaraderie, rapport, kindness

Energetic/Outgoing

outgoing, energetic, lively, enthusiastic, animated, vibrant, vivacious, spirited, bubbly, dynamic, energy, enthusiasm, vivacity, vibrancy, dynamism, zest, gaiety, exuberance, animation, zestfulness

Cooperativeness

cooperative, helpful, supportive, collaborative, considerate, accommodating, flexible, team-oriented, empathetic, understanding, cooperation, teamwork, collaboration, support, flexibility, accommodation, empathy, coordination, synergy, unity

Humor

humorous, witty, funny, jovial, entertaining, amusing, lighthearted, playful, comical, cheerful, laugh, giggle, joke, humor, wit, levity, jocularity, whimsy, playfulness, banter, hilarity, satire, quip

Activities/Engagement

Athletic Participation

athletic, sporty, agile, fit, strong, robust, vigorous, active, nimble, competitive, athletics, sports, competition, training, fitness, endurance, agility, marathon, sprint, match, track, ball, basketball, football, run

Arts/Cultural Activities

artistic, musical, creative, talented, expressive, cultured, aesthetic, inventive, imaginative, visionary, painting, music, dance, theater, sculpture, literature, exhibition, concert, festival, gallery

Academic Involvement

studious, scholarly, intellectual, educated, knowledgeable, analytical, learned, erudite, literate, bookish, research, scholarship, lecture, seminar, dissertation, analysis, journal, conference, thesis, coursework

Club/Service Participation

involved, engaged, active, committed, enthusiastic, volunteer, service-oriented, civic-minded, helpful, debate, committee, volunteer, outreach, fundraiser, workshop, council, society, association, club

### other

can year be used?

## 250522

llm repeat

placebo test
