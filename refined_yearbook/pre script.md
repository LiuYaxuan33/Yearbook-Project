
**Title Slide**  
Good afternoon, everyone.  
My name is Liu Yaxuan, from Yuanpei College.  
Today I’ll be presenting my project, ‘Gendered Peer Evaluations: Text Analysis of Early 20th Century U.S. College Yearbooks.
Let’s get started.

---

**Slide 2: Outline**  
Let me begin with a brief outline of today’s talk. I will first introduce the research background and motivation, followed by an overview of the data and selected samples. Then I will describe the methodology, including the use of natural language processing techniques. After that, I will share some preliminary results, both from word feature extraction and sentiment analysis. Finally, I will conclude with our main findings and plans for future work.

---

**Slide 3: Background and Motivation**  

The starting point of this research is the observation that yearbooks provide a rare bottom-up perspective on peer-to-peer evaluations.  
While recent narrative studies have flourished, they mostly focus on top-down evaluations—such as course grades, recommendation letters, Twitter posts, or news reports. Peer-to-peer assessments, written spontaneously by students, are much less examined.  
Alongside the growth of interpretable NLP techniques—like word frequency analysis, regularized regression,  or say naive machine learning models, and sentiment lexicons—there is now a unique opportunity to explore these peer narratives systematically.  
Our central research questions are: How is gender bias reflected in peer comments? And do men’s and women’s comments differ systematically in emotional tone?


---

**Slide 4: Data Description**  


Turning to the data, we compiled graduation profiles from Iowa State University yearbooks, specifically for the years 1906, 1909, and 1911 through 1916, because they contain comments made by student committee that are long enough.  
We used OCR models PaddleOCR and QWen to extract each graduate’s name, hometown, club membership,  the full comment text and other information, if any. 
The students' gender is inferred from name and photo; and in cases of mismatch we manually checked the portrait to assign gender.  Due to limited time I won't introduce by detail the pipeline in constructing the dataset, and you are welcomed to ask for clarification.
In total, we obtained 2,120 samples, including 365 female profiles.

---

**Slide 5: Sample Records: Stereotypes in Action**  

To give you a flavor of the material, here are two real examples.  
One profile describes a Chinese student as, quote, “A hard and accurate worker. Found out why the adding machine made mistakes.”  
This reflects a racial stereotype, framing Chinese students as diligent technically competent, and good at math, but implicitly, perhaps not innovative.  
Another profile reads, “She has shown that the happiest women, like the happiest nations, have a history,” referring to a female graduate.  
You can also see, under the guise of admiration, hints at gossip regarding her private life.  
Such examples vividly show the presence of ethnic and gender stereotypes in peer narratives of that era.


**Slide 6: Research Design**  

Our research design follows three steps.  
First, we digitized the yearbooks and constructed a completely novel dataset through a combination of OCR extraction, LLM-assisted text polishing, and manual proofreading.  
Second, we applied Lasso regression to identify words and phrases most predictive of gender.  
Third, we conducted sentiment analysis using two complementary approaches: one based on TF-IDF weighted lexicons, and the other relying on direct scoring via a large language model, DeepSeek v3.


**Slide 7: Lasso Regression: Principle**  
Let me now briefly explain Lasso regression. It applies an L1 penalty, in addition to ordinary least squares:  
The L1 penalty forces many coefficients to zero, selecting only the strongest predictors.  As you can see from the graph, because of the absolute value form of the penalty term, it encourages the optimal point of parameters to be sparse. Here, our predictors \(X\) are word dummy features for terms appearing at least five times; the outcome \(y\) is binary gender, 1 for female and 0 for male.  The nonzero coefficients, selected by Lasso, highlight the most gendered vocabulary.


---

**Slide 8: Key Words Identified by Lasso**  
Here is the Lasso output for unigrams, and stop-words are previously deleted from the word pool.  
On the right are words with positive coefficients—terms more associated with females, such as "calm", "lovable", "winsome", "gentle", "Beautiful", and interestingly, "democratic".  These are words that are judging, especially used to judge girls in traditional idea.
On the left are male‑leaning words, though not as gendered, but still some interesting words, like "athletic", "chief", "able".
Though this clear separation aligns with common stereotypes, there are some words that are quite hard to understand, such as "hot" for males. By delving into the original data, we found that the word "hot" mostly co-appear with "air", and the expression"hot air" actually means empty talk. This led me to take 2-grams into consideration. As the former literature usually did, we search for 2-grams by first deleting the stop words and then look for word pairs that appear successively. 


---

**Slide 9: 2-grams Identified by Lasso**  
In the 2-gram results, gendered language became even clearer. Here we have "little girl", which is kind of frivolous when used to talk about girls, and appearance descriptions like "blue eyes", and personality phrases like "loyal friend", "big heart" and "work hard". For males we have "good fellow", "hot air" as previously mentioned, "practical work", and "rough housing", a phrase used to refer to rough and wild play or fighting.



---

**Slide 10: Examples of Gendered Language in Context**  
We have two examples of gendered words here. One is adapted from a girl that graduated in 1912, and her comment says, "Joys are our wings; sorrows our spurs. Winsome, merry little Polly, liked by all who know her.……Has proved her ability to manage a household."We see winsome and merry here, and it talks about her ability in managing a household, as an indicator of gender norm. For the boy we have, interestingly, "big-hearted", which is statistically for girls in our data. Also we have "practical" here, and you can see "he was equal to all occasions by the way he fooled the hazers in those good old day", talking about his ability.


---

**Slide 11: Two Approaches to Sentiment Analysis**  
Now we enter the section of sentiment analysis. Here, we adopted two parallel approaches. 
The method we used first involves using of Empath, a lexicon of emotions and topics developed by Stanford.
It has more than two hundred categories of topics and emotions, from which we selected eight dimensions that is suitable for our corpus. They are affection, trust, achievement, help, work independence, positive and negative emotions. As examples, here are words of two categories, just listed to show how this lexicon works.
Our first method uses the Stanford Empath lexicon, and computed TF‑IDF scores for each dimension’s word list, then weight and sum, to calculate the score of each dimensions. The tf-idf method is widely accepted and used in word-frequency-based NLP tasks. Its intuition is, on the basis of naive word frequency, in order to take into consideration the inherit frequency of a certain word among all the documents, we add the modify factor in logarithm. And it is said that it works not only much faster, but also amazingly accurate in tasks where the scale of data is limited.
The second method  calls the DeepSeek v3 API with a prompt like this and asks the LLM to rate each comment from 0 to 1 on those same eight dimensions. The reason we used deepseek instead of other llms is that its cheap, and it could be interesting to compare between different models, though we are not going to do it.
In the appendix, we gave the consistency check of the two methods, and they are somehow consistent, but deepseek seems to have done better in understanding complicated meanings and sentiments.


---

**Slide 12: Sentiment by Gender: Radar Chart**  
Here first is the radar chart of lexicon-based sentiment score. We plot male, which is the orange line, and female, which is blue line.Only dimensions scored higher than 0.04 are showed. We can see, for example, female scores higher in ”friends”; males in ”masculine”, which seems quite reasonable.
And here I listed radar charts of lexicon and llm score result.
Notice women score higher on Affection and Positive Emotion, help and trust, while men are higher on work and Achievement.  
Crucially, both Empath and DeepSeek agree on these dimensions, reinforcing robustness.


---

**Slide 13: Examples of sentiment scoring**  
Now we have some examples here, and you can see that, as we discussed, deepseek has done well in understanding metaphors and sarcasm, as showed here "Might draw highest honors of her class if her head were not so filled with Paine." And in the examples 3 and 4 we see words that falls in particular dimensions, and you can see how the lexicon-based method works.


---

**Slide 14: Conclusion and Next Steps**  
To summarize, we managed to specify gendered word use in peer yearbook comments and tried to quantify Emotional tone likewise diverges by gender across multiple dimensions. Our mixed‑method NLP strategy—combining Lasso, lexicon, and LLM—yields consistent insights, and contributes to literature both in methodology of text analysis and in narrative studies on peer-focused comments.  

As you may realized, the research up to now still needs perfection. For future work, we found it actually important to reconsider the dimension selection in sentiment analysis, perhaps with reference to psychology literatures, which we are sorry to fail to take into account, due to limited time . At last, we will extend analysis by academic major and perhaps hometown to see how context shapes peer language. Also, it may be rewarding to try different and advanced NLP algorithms, such as word-embedding, cosine similarity and probably RAG(retrieval-augmented generation), if possible. More importantly, we will try to make good use of LASSO-selected word features for further exploration, for example classification and again, calculate their frequencies, as you may have noticed that so far our research is still somehow separated, and the lasso part and sentiment part have few connections.


---

**Slide 15: Conclusion and Next Steps**  
In the last it is important to note the sources that offered help to this research, and I would like to specially thank  Fu Minshan for helping in constructing dataset. The references are also listed here, including introductions to text analysis method used in economics research, and also relevant studies involving text, sentiment and gender bias analysis.



---

**Slide 16: Thank You!**  
And that's all for my presentation, thank you for listening, and questions are welcomed. In the appendix I listed some regression results, goodness of fitness of LASSO, t-test for dimensional sentiment score between genders, and consistency test of the two parallel methods of sentiment analysis. It may be important to note that the sentiment scores using lexicon-based method between dimensions are completely independent, that is to say, the relation between them is not like control variables, and whether taking or not into account of other dimensions will not influence scores got in this dimension.