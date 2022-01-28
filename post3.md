# Using Machine Learning and Natural Language Processing tools for text analysis

This is a third article on the topic of guided projects feedback analysis. The main idea of the topic is to analyse the responses students are receiving on the forum page. Dataquest encourages its students to publish their guided projects on their forum, after publishing other students or staff members can share their opinion of the project. We're interested in the content of those opinions.

In our [previous post](https://www.dataquest.io/blog/how-to-clean-and-prepare-your-data-for-analysis/) we've done a basic data analysis of numerical data and dove deep into analysing the text data of feedback posts. 


In this article, we'll try multiple packages to enhance our text analysis. Instead of setting a goal of one task, we'll play around with various tools that use natural language processing and/ or machine learning under the hood to deliver the output. 

Here's a list of goodies we'll try in this article:
* sentiment analysis
* keyword extraction
* topic modelling 
* kmeans clustering
* concordance
* query answering model



Using machine learning for NLP is a very broad topic and it's impossible to contain it within one article. You may find that the tools described in this article are not important from your point of view. Or that they've been used incorrectly, most of them were not adjusted, we've just used out of the box parameters. Remember it is a subjective selection of packages, tools and models that had been used for enhancing the analysis of feedback data. But you may be able to find a better tool or a different approach. We encourage you to do so and share your findings. 


# ML VS NLP
Machine learning and Natural Language Processing are two very broad terms that can cover the area of text analysis and processing. We're not going to try to set a fixed line between these two terms, we'll leave that to the philosophers. If you're interested in pursuing the difference between them look [here](https://www.projectpro.io/article/machine-learning-vs-nlp/493).

# Sentiment analysis

Sentiment analysis is a very popular application of ML power. By analyzing the content of each text we can evaluate how positive or negative the weight of the sentence or the whole text is. This can be of a huge value if you want to filter out the negative reviews of your product or present only the good ones. 

There are tons of sentiment analysis models and tools for python available online. We'll focus on one of the simplest ones: it will take us 2 lines of code to perform a basic sentiment analysis:
```python
# import the package:
from pattern.en import sentiment
# perform the analysis:
x = 'project looks amazing, great job'
sentiment(x)
```
> Output:
```
(0.7000000000000001, 0.825)
```
As seen above after importing the package we just have to call the sentiment function, provide it with a string value and Voila! The output is a tuple with the first value being the 'polarity' (how positive the sentence is on a scale from -1 to 1). The second value is the subjectivity which tells us how certain the algorithm is about the assessment of its first value, this time the scale starts at 0 and ends at 1. Let's have a look at a few more examples:

```python
y = 'plot looks terrible, spines are too small'
sentiment(y)
```
> Output:
```
(-0.625, 0.7)
```
We're being served with a rather 'low' first value, and the algorithm is still quite confident about its polarity assessment. Let's try something harder:

```python
z = 'improve the comment, first line looks bad'
sentiment(z)
```
> Output:
```
(-0.22499999999999992, 0.5)
```
We can notice that the model is more hesitant on this one. The string has a negative word, but it's not so sure about it anymore.

Let's apply the sentiment function to our feedback content:

```python
df['sentiment'] = df['feedback_clean2_lem'].apply(lambda x: sentiment(x))
df['polarity'] = df['sentiment'].str[0]
df['subjectivity'] = df['sentiment'].str[1]
df = df.drop(columns='sentiment')
```
Now, that we have measured the polarity and subjectivity of each feedback post, let's inspect how each topic differs:


```python
top15 = df['short_title'].value_counts()[:15].index
df[df['short_title'].isin(top15)].groupby('short_title')[['polarity','subjectivity']].mean().sort_values('subjectivity')
```

| short_title         |   polarity |   subjectivity |
|:--------------------|-----------:|---------------:|
| sql using           |   0.227535 |       0.441175 |
| cia factbook        |   0.232072 |       0.460941 |
| car prices          |   0.206395 |       0.476976 |
| ebay car            |   0.251605 |       0.498947 |
| traffic heavy       |   0.219977 |       0.504832 |
| wars star           |   0.250845 |       0.50871  |
| news hacker         |   0.288198 |       0.509783 |
| exit employee       |   0.269066 |       0.512406 |
| science popular     |   0.276232 |       0.514718 |
| app profitable      |   0.281144 |       0.514833 |
| nyc high            |   0.288988 |       0.519288 |
| fandango ratings    |   0.265831 |       0.524607 |
| gender gap          |   0.285667 |       0.534309 |
| college visualizing |   0.279269 |       0.547273 |
| markets advertise   |   0.279195 |       0.572073 |

Unfortunately, the results are very similar. It comes as no surprise, most of the feedback posts have a very similar structure. They usually contain a sentence or two congratulating on the project at first. This positive content is usually followed by some critical remarks (usually treated as content with negative polarity). The post usually ends with some positive message for future coding. In essence, it's an absolute mess of intertwined messages of positive and negative sentiment. Not as easy as product reviews where very often we come across a happy client or a very unhappy one. Categorizing their reviews is not that hard most of the time. Unfortunately for us, our content is more complex. But we won't give up so easily.



# Keywords

Extracting keywords from a given string is another hefty trick, that can improve our analysis. 

Rake package delivers a list of all the n-grams and their weight extracted from the text. The higher the value, the more important is the n-gram being considered. After parsing the text, we can filter out only the n-grams with the highest values. 

Be aware though, **the model is using stopwords in assessing which words are important within the sentences. If we were to feed this model with a text cleaned of stopwords, we wouldn't get any results.**

```python
from rake_nltk import Rake
# set the parameteres (length of keyword phrase):
r = Rake(include_repeated_phrases=False, min_length=1, max_length=3)
text_to_rake = df['feedback'][31]
r.extract_keywords_from_text(text_to_rake)
# filter out only the top keywords:
words_ranks = [keyword for keyword in r.get_ranked_phrases_with_scores() if keyword[0] > 5]
words_ranks
```

>Output:
```
[(9.0, '“ professional ”'),
 (9.0, 'avoiding colloquial language'),
 (8.0, 'nicely structured project'),
 (8.0, 'also included antarctica'),
 (8.0, 'add full stops')]
```

In this example, Rake decided that 'professional' or 'avoiding colloquial language' are the most important keywords of the input text. For further analysis, we won't be interested in the numerical values of the keywords. We just want to receive a few top keywords for each post. We'll design a simple function for extracting only the top keywords and apply it to the 'feedback' column:

```python
def rake_it(text):
    r.extract_keywords_from_text(text)
    r.get_ranked_phrases()
    keyword_rank = [keyword for keyword in r.get_ranked_phrases_with_scores() if keyword[0] > 5]
    # select only the keywords and return them:
    keyword_list = [keyword[1] for keyword in keyword_rank]
    return keyword_list

df['rake_words'] = df['feedback'].apply(lambda x: rake_it(x))
```

Having extracted the keywords from every post, let's see which ones are the most popular! Remember they are being stored as a list inside a cell, so we have to deal with that obstacle:

```python
# function from: towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

to_1D(df['rake_words']).value_counts()[:10]
```
>Output:
```
jupyter file menu        24
guided project use       24
happy coding :)          22
everything looks nice    20
sql style guide          16
first guided project     16
everything looks good    15
new topic button         15
jupyter notebook file    14
first code cell          13
dtype: int64
```
# Topic modelling

Topic modelling can quickly give us an insight into the content of the text. Unlike extracting keywords from the text, topic modelling is a much more advanced tool that can be tweaked to our needs.

We're not going to venture too deep into designing and implementing this model, that itself can fill out a few articles. We're just going to quickly run the basic version of this model on each feedback content. We aim to extract a designated number of topics for each post. 

We're going to start with one feedback post. Let's import the necessary packages, compile the text and create the required dictionaries and matrixes(it's machine learning so each model requires a specific input):

```python
# Importing:
import gensim
from gensim import corpora
import re

# compile documents, let's try with 1 post:
doc_complete = df['feedback_clean2_lem'][0]
docs = word_tokenize(doc_complete)
docs_out = []
docs_out.append(docs)

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(docs_out)

# Converting a list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_out]
```

Having done all the preparation work, it's time to train our model and extract the results. As you may have already noticed, we can manually set a lot of parameters. To name a few: the number of topics, how many words we use per topic. But the list goes on and as in the case of many ML models, you can spend a lot of time tweaking those parameters to perfect your model.


```python
# Running and Trainign LDA model on the document term matrix.
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50, random_state=4)

# Getting results:
ldamodel.print_topics(num_topics=5, num_words=4)
```
>Output:
```
[(0, '0.032*"notebook" + 0.032*"look" + 0.032*"process" + 0.032*"month"'),
 (1, '0.032*"notebook" + 0.032*"look" + 0.032*"process" + 0.032*"month"'),
 (2, '0.032*"important" + 0.032*"stay" + 0.032*"datasets" + 0.032*"process"'),
 (3,
  '0.032*"httpswww1nycgovsitetlcabouttlctriprecorddatapage" + 0.032*"process" + 0.032*"larger" + 0.032*"clean"'),
 (4, '0.113*"function" + 0.069*"inside" + 0.048*"memory" + 0.048*"ram"')]
```

We can notice that our model is providing us with a list of tuples, each tuple contains words and their weight. The higher the number, the more important the word (according to our model). If we wanted to dive deep we could extract all the words above a certain value... Just an idea :) 

Let's move on to applying the model to each feedback post. To simplify our lives, we'll extract only the topic words, not the 'weight' values. That way, we can easily perform value_counts on extracted topics and see what topics were the most popular (according to the model). To perform topic modelling on each cell in the column, we'll design a function. As input values we'll use the content of the cell (text) and the number of words for the topic:

```python
def get_topic(x,n):
    """
    extract list of topics given text(x) and number(n) of words for topic
    """
    docs = word_tokenize(x)
    docs_out = []
    docs_out.append(docs)
    dictionary = corpora.Dictionary(docs_out)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_out]
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50, random_state=1)
    topics = ldamodel.print_topics(num_topics=2, num_words=n)
    topics_list = []
    for elm in topics:
        content = elm[1]
        no_digits = ''.join([i for i in content if not i.isdigit()])
        topics_list.append(re.findall(r'\w+', no_digits, flags=re.IGNORECASE))
    return topics_list
    
```
Let's see how topics with a maximum length of 4 words look:
```python
df['topic_4'] = df['feedback_clean2_lem'].apply(lambda x: get_topic(x,4))
to_1D(df['topic_4']).value_counts()[:10]
```
>Output:
```
[keep, nice]                                               8
[nice, perform]                                            4
[nice]                                                     4
[nan]                                                      4
[sale, take, look, finish]                                 2
[learning, keep, saw, best]                                2
[library, timezones, learning, httpspypiorgprojectpytz]    2
[job, graph, please, aesthetically]                        2
[thanks, think, nice, learning]                            2
[especially, job, like, nice]                              2
dtype: int64
```
And now, let's check maximum 3-worded topics:
```python
df['topic_3'] = df['feedback_clean2_lem'].apply(lambda x: get_topic(x,3))
to_1D(df['topic_3']).value_counts()[:10]
```

>Output:
```
[keep, nice]                       8
[share, thank, please]             4
[nan]                              4
[nice]                             4
[nice, perform]                    4
[guide, documentation, project]    3
[]                                 3
[cell]                             3
[guide, project, documentation]    3
[plot]                             3
dtype: int64
```

Our results are not great, but remember we've just scratched the surface of this tool. There's a lot of potential when it comes to lda model. I encourage you to read at least one of the below articles, to broaden your understanding of this library:  

[analyticsvidhya](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)

[machinelearningplus](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)

[towardsdatascience](https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925)


# K-means clustering

Kmeans model can cluster the data based on various inputs, it's probably the most popular unsupervised machine learning model. Just select how many clusters do you want the data to be assigned to, based on what features and Voila. Being a ML model we can't just feed it with raw text, we have to vectorize the text data and then feed the model with it. In essence, we're transforming the text data into numeric data. How we do it is up to us, there are many ways to vectorize the data, let's try TfidfVectorizer:

```python
# imports:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# vectorize text:
tfidfconverter = TfidfVectorizer(max_features=5000, min_df=0.1, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(df['feedback']).toarray()

# fit and label:
Kmean = KMeans(n_clusters=8, random_state=2)
Kmean.fit(X)
df['label'] = Kmean.labels_
```
 Now, let's see if different clusters have a large polarity difference:
 ```python
df.groupby('label')['polarity'].mean()
 ```
 >Output:
 ```
label
0    0.405397
1    0.312328
2    0.224152
3    0.210876
4    0.143431
5    0.340016
6    0.242555
7    0.241244
Name: polarity, dtype: float64
 ```
 Actually, let's check a few other numbers based on the cluster number assigned by our model:
 ```python
 # group data:
polar = df.groupby('label')['polarity'].mean()
subj = df.groupby('label')['subjectivity'].mean()
level = df.groupby('label')['level'].mean()
val_cnt = df['label'].value_counts()
length = df.groupby('label')['len'].mean()

# create a df and rename some of the columns, index:
cluster_df = pd.concat([val_cnt,polar,subj,level, length], axis=1)
cluster_df = cluster_df.rename(columns={'label':'count'})
cluster_df.index.name = 'label'
cluster_df
 ```

|   label |   count |   polarity |   subjectivity |   level |      len |
|--------:|--------:|-----------:|---------------:|--------:|---------:|
|       0 |      87 |   0.405397 |       0.635069 | 1.49425 |  314.287 |
|       1 |     150 |   0.312328 |       0.536363 | 1.55333 |  742.253 |
|       2 |      60 |   0.224152 |       0.469265 | 1.5     |  594.267 |
|       3 |     136 |   0.210876 |       0.513048 | 1.46324 | 1429.1   |
|       4 |      66 |   0.143431 |       0.34258  | 1.4697  |  251.227 |
|       5 |     118 |   0.340016 |       0.581554 | 1.29661 |  903.11  |
|       6 |     302 |   0.242555 |       0.495008 | 1.45033 |  724.209 |
|       7 |      92 |   0.241244 |       0.431905 | 1.52174 |  398.228 |

 We can notice some interesting trends in this table, eg. cluster number 0 has a rather positive content(high polarity mean), also the sentiment model we've used before in this cluster is rather certain about its modelling (high subjectivity value). This may be caused by a very short average length of text in this cluster (314). Have you found any other interesting facts looking at the above table?
 
Remember that we've fed the Kmeans model with a data vectorized with Tfidf, there are [multiple ways](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/) of [vectorizing](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) text data before feeding it to a model. You should try them and check how they affect the results.



# Clustering sentences

As mentioned before: the content of each feedback post is a rather complicated mix of compliments and constructive criticism. That's why our clustering model didn't perform well when asked to cluster posts. But if we were to split all of the posts into sentences and ask the model to cluster sentences, we should improve our results.

```python
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk import sent_tokenize
from nltk import pos_tag
import string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# set up for cleaning and sentence tokenizing:
exclude = set(string.punctuation)
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    return punc_free

# remember this function from article no 2?
def lemmatize_it(sent):
    empty = []
    for word, tag in pos_tag(word_tokenize(sent)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
            empty.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, wntag)
            empty.append(lemma)
    return ' '.join(empty)


# tokenize sentences and clean them:
doc_complete = sent_tokenize(df['feedback'].sum())
doc_clean = [clean(doc) for doc in doc_complete] 
doc_clean_lemmed = [lemmatize_it(doc) for doc in doc_clean] 

# create and fill a dataframe with sentences:
sentences = pd.DataFrame(doc_clean_lemmed)
sentences.columns = ['sent']
sentences['orig'] = doc_complete
sentences['keywords'] = sentences['orig'].apply(lambda x: rake_it(x))

# vectorize text:
tfidfconverter = TfidfVectorizer(max_features=10, min_df=0.1, max_df=0.9, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(sentences['sent']).toarray()

# fit and label:
Kmean = KMeans(n_clusters=8)
Kmean.fit(X)
sentences['label'] = Kmean.labels_
```
Now let's check the most popular keywords for each label:

```python
to_1D(sentences[sentences['label'] == 0]['keywords']).value_counts()[:10]
```

>Output:
```
everything looks nice     15
everything looks good     13
sql style guide           10
print () function          8
social love attention      7
data cleaning process      7
looks pretty good          6
screen shot 2020           5
everything looks great     5
jupyter notebook file      5
dtype: int64
```

```python
to_1D(sentences[sentences['label'] == 2]['keywords']).value_counts()[:10]
```
>Output:
```
first code cell            8
avoid code repetition      7
items (): spine            4
might appear complex       4
may appear complex         4
might consider creating    3
1st code cell              3
print () function          3
little suggestion would    3
one code cell              3
dtype: int64
```


# Clustering n-grams

Similar to clustering posts and sentences we can perform clustering of n-grams:

```python
from nltk.util import ngrams 
import collections

# extract n-grams and put them in a dataframe:
tokenized = word_tokenize(sentences['sent'].sum())
trigrams = ngrams(tokenized, 3)
trigrams_freq = collections.Counter(trigrams)
trigram_df = pd.DataFrame(trigrams_freq.most_common())
trigram_df.columns = ['trigram','count']
trigram_df['tgram'] = trigram_df['trigram'].str[0]+' '+trigram_df['trigram'].str[1]+' '+trigram_df['trigram'].str[2]

# vectorize text:
tfidfconverter = TfidfVectorizer(max_features=100, min_df=0.01, max_df=0.9, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(trigram_df['tgram']).toarray()

# fit and label:
Kmean = KMeans(n_clusters=8, random_state=1)
Kmean.fit(X)
trigram_df['label'] = Kmean.labels_
```

# Concordance


https://avidml.wordpress.com/2017/08/05/natural-language-processing-concordance/

What if we want to check how a specific word is being used inside a text? We'd like to look at the words before and after that specific word. With a little help from concordance, we can quickly have a look:

```python
from nltk.text import Text 
text = nltk.Text(word_tokenize(df['feedback'].sum()))
text.concordance("plot", lines=10)
```
>Output:
```
Displaying 10 of 150 matches:
ive precision . it is better to make plot titles bigger . about the interactiv
 '' ] what is the point of your last plot ? does it confirm your hypothesis th
ou use very similar code to create a plot – there is an opportunity to reduce 
er plots ” try to comment after each plot and not at the end so the reader doe
ur case saleprice , and after it you plot correlation only between remaining f
tation then possible and correlation plot may be have different colors and val
tting the format of the grid on your plot setting the ylabel as ‘ average traf
 line_data = series that you want to plot # start_hour = beginning hour of the
# start_hour = beginning hour of the plot **in 24hr format** # end_hour = end 
ormat** # end_hour = end hour of the plot **in 24hr format** def plot_traffic_
```

# Query answering model

Let's be honest to start a simple query answering model you don't need to understand a lot about the specific mechanics of the witchcraft happening inside the model. It's worth knowing the basics:
* you have to transform the text into vectors and arrays
* the model compares that numerical input and finds a content that is the most similar to the input
* that's it, after you've managed to run a basic model, you should start experimenting with different parameters and vectorization methods  


```python
# imports:
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stopword_list = stopwords.words('english')

# create a model:
dtm = CountVectorizer(max_df=0.7, min_df=5, token_pattern="[a-z']+", 
                      stop_words=stopword_list, max_features=6000)

dtm.fit(df['feedback_clean2_lem'])
dtm_mat = dtm.transform(df['feedback_clean2_lem'])
tsvd = TruncatedSVD(n_components=200)
tsvd.fit(dtm_mat)
tsvd_mat = tsvd.transform(dtm_mat)

# let's look for "slow function solution"
query = "slow function solution"
query_mat = tsvd.transform(dtm.transform([query]))

# calculate distances:
dist = pairwise_distances(X=tsvd_mat, Y=query_mat, metric='cosine')
# return the post with the smallest distance:
df['feedback'][np.argmin(dist.flatten())]
```
>Output:
```
' processing data inside a function saves memory (the variables you create stay inside the function and are not stored in memory, when you are done with the function) it is important when you are working with larger datasets - if you are interested with experimenting: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page try cleaning 1 month of this dataset on kaggle notebook (and look at your ram usage) outside the function and inside the function, compare the ram usage in both examples '
```

Remember that the dataset we're parsing to look for an answer is rather small, so we can't expect mind-blowing answers. 

# Last word

This is not the end of a very long list of tools used for text analysis. We've barely scratched the surface and the tools we've used haven't been used most efficiently. You should continue and look for a better way, tweak that model, use a different vectorizer, gather more data.

The crazy mix of Natural Language Processing and Machine Learning is a never-ending topic that can be studied for decades. Just the last 20 years have brought us amazing applications of these tools, do you remember the world before Google? When searching content on the internet was very similar to looking at yellow pages? How about using our smartphone assistants? Those tools are constantly getting more efficient, it's worth directing your attention to how are they becoming better at understanding our language.

## Any questions?

Feel free to reach out and ask me anything:
[Dataquest](https://community.dataquest.io/u/adam.kubalica/summary), [LinkedIn](https://www.linkedin.com/in/kubalica/), [GitHub](https://github.com/grumpyclimber/portfolio)
