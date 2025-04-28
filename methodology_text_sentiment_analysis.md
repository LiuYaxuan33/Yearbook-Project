

Existing literatures introduced NLP method used in economics. Dictionary-based method require pre-defined dictionaries of sentiment-related words and count their occurrences in documents(Ash, E. et al., 2023; Shapiro, A. H. et al., 2020). As an example, Barbaglia et al.(2025) constructed Economic Lexicon with modifier words extracted from newspaper text and annotated by human experts, and found it behaving well in forecasting economic recessions. Other word-based methods include LDA, tf-idf, etc. Some need pre-specified topics such as Latent Dirichlet Allocation (LDA), and some don't, such as Principal Component Analysis (PCA) and Latent Semantic Analysis (LSA), for they (Loosely speaking) are just maximizing variance.

Another series of method involve vector representation(embedding) of words and often deep learning technologies. Representation of words can come from pre-trained language models, for example word2vec, or specially trained model for specific application scenario, but only when the data set is sufficiently large. Word embedding can also be used to expand dictionaries mentioned above, based on calculating cosine similarity between new words and seed words. Deep learning techniques can be supervised (requiring manually labeled data) or self-supervised, but both need substantial amount of data and computational resources(Shapiro, A. H. et al., 2020; Ash, E. et al., 2023). Affordable usage of deep learning method may be fine-tuning existing transformer models, but from practical experience such method can barely exceed prompt engineering, i. e. calling online LLM API. Though this may be useful in data annotation, lacks interpretability of output(Ash, E. et al., 2023; Shapiro, A. H. et al., 2020) and commonly accepted standards in prompt engineering(Ash, E. et al., 2023).

To sum up, word-based method seems to be still suitable to analyze sentiment in comments on students. As one of examples of usage of such method, Song et al.(2019) construct the NCSI using the lexicon approach, manually creating a Korean sentiment lexicon and calculating weighted sum of frequency of words with different emotion scores(0, 1 and -1). Similar methods used in Finance are combined with time series analysis(Kearney, C. et al., 2014).

#### Reference

Ash, E., & Hansen, S. (2023). Text Algorithms in Economics. *Annual Review of Economics*, 15, 659–688. https://doi.org/10.1146/annurev-economics082222-074352

Barbaglia, L., Consoli, S., Manzan, S., Tiozzo Pezzoli, L., & Tosetti, E. (2025). Sentiment analysis of economic text: A lexicon‐based approach. *Economic Inquiry*, 63(1), 125–143. https://doi.org/10.1111/ecin.13264

Song, M., & Shin, K. (2019). Forecasting economic indicators using a consumer sentiment index: Survey‐based versus text‐based data. *Journal of Forecasting*, 38(5), 504–518. https://doi.org/10.1002/for.2584

Shapiro, A. H., Sudhof, M., & Wilson, D. J. (2020). Measuring news sentiment. Journal of Econometrics, 228, 221 - 243.

Kearney, C., & Liu, S. (2014). Textual sentiment in finance: A survey of methods and models. *International Review of Financial Analysis*, 33, 171-185. https://doi.org/10.1016/j.irfa.2014.02.006