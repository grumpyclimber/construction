In this article we'll try multiple packages to enhance our text analysis. Instead of setting a goal of one task, we'll play around with various tools that use machine learning and or natural language processing under the hood to deliver the output. 

Using machine learning for NLP is a very broad topic and it's impossible to contain it within one article. You may find that the tools described in this article are not important from your point of view. Or that they've been used incorrectly, most of them were not tweaked properly, we've just used out of the box parameters. Rememeber it is a subjective selection of packages, tools and models that had been used for enhancing the analysis of feedback data. But you may be able to find a better tool or a different approach. We encourage you to do so and share your findings. 


# ML VS NLP
Machine learning and Natural Language Processing are two very broad terms that can cover the same areas of text analysis and processing. We're not going to try to set a fixed line between these two terms, we'll leave that to the philosophers. If you're interested in pursuing the difference between them look here: 
https://www.projectpro.io/article/machine-learning-vs-nlp/493

# Sentiment analysis

Sentiment analysis is a very popular application of ML power. By analyzing the content of each text we can evaluate how positive or negative the weight of the sentence or the whole text is. This can be of a huge value if you want to filter out the negative reviews of your product, or present only the good ones. 

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
As seen above after importing the package we just have to call the sentiment function, provide it with a string value and Voila! The output is a tuple with the first value being the 'sentiment' (how positive the sentence is on a scale from -1 to 1). The second value is the subjectivity which tells us how certain the algorithm is about its first values assesment, this time the scale starts at 0 and ends at 1. Let's have a look at a few more examples:

```python
y = 'plot looks terrible, spines are too small'
sentiment(y)
```
> Output:
```
(-0.625, 0.7)
```
We're being served with a rather 'low' first value, and the algorithm is still quite confident about it's polarity assesment. Let's try something harder:

```python
z = 'improve the comment, first line looks bad'
sentiment(z)
```
> Output:
```
(-0.22499999999999992, 0.5)
```
We can notice that this model is hesitant on this one. The string has a negative word, but it's not so sure about it anymore.

Let's apply the sentiment function to our feedback content:

```python
df['sentiment'] = df['feedback_clean2_lem'].apply(lambda x: sentiment(x))
df['polarity'] = df['sentiment'].str[0]
df['subjectivity'] = df['sentiment'].str[1]
df = df.drop(columns='sentiment')
```
Now, that we have meassured polarity and subjectivity of each feedback post, let's inspect how each topic differs:


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

Unfortunatelly, the results are very similar. FINISH

BLABLA



# Keywords

Extracting keywords from a given string is another hefty trick, that can improve our analysis. 

Rake package delivers a list of all the n-grams and their weight extracted from the text. The higher the value, the more important is the n-gram being considered. After parsing the text, we can filter out only the n-grams with the highest values. 

Be aware though, **the model is using stopwords in assesing which words are important within the sentences. If we were to feed this model with a text cleaned of stopwords, we wouldn't get any results.**

```python
from rake_nltk import Rake
# set the parameteres:
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

In this example Rake decided that 'professional' or 'avoiding colloquial language' are the most important keywords of the input text. For the purpose of further analysis we won't be interested in the numerical values of the keywords. We just want to receive a few top keywords for each post. We'll design a simple function for extracting only the top keywords and apply it to 'feedback' column:

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

# concordance

```python
from nltk.text import Text 
text = nltk.Text(word_tokenize(df['feedback'].sum()))
text.concordance("plot", lines=10)
```
>Output:
```
Displaying 15 of 150 matches:
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
e fivethirtyeight style for the last plot . why not use the same style for the
he categories on the “ service_cat ” plot in a logical way ( from “ new ” to “
 amazing ! i love your angle and the plot gif you made is obviously beyond exp
ly beyond expectation . on the first plot i also really like how you show whic
uickly absorb the information in the plot gif . your code is really clean and 
```
# TFIDF

# topic modeling

Topic modeling can quickly give us an insight into the content of the text. Unlike extracting keywords of the text, topic modeling is a much more advanced tool that can be tweaked to our needs.

We're not going to venture too deep into desinging and implementing this model, that itself can fill out a few articles. We're just going to quickly run the basic version of this model on each feedback content. Our aim is to extract a designated number of topics for each post. 

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

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_out]
```

Having done all the preperation work, it's time to train our model and extract the results. As you may have already noticed, we can manually set a lot of parameters. To name a few: the number of topics, how many words we use per topic. But the list goes on and as in case of many ML models, you can spend a lot of time tweaking those parameters to perfect your model.


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

We can notice that our model is providing us with a list of tuples, each tuple contains words and their weight. The higher the number, the more important the word (according to our model). If we wanted to dive really deep we could extract all the words above a certain value... Just an idea :) 

Let's move on to applying the model to each feedback post. To simplify our lifes, we'll extract only the topic words, not the 'weight' values. That way, we can easiy perform value_counts on extracted topics and see what topics were the most popular (according to the model). To perform topic modeling on each cell in the column, we'll design a function. As input values we'll use the content of the cell (text) and number of words for topic:

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
Let's see how 4-worded topics look:
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
And now, let's check 3-worded topics:
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



# query answer

# Similarity

# k means clustering

