# UNDER CONSTRUCTION:

# Title

Do you share your projects in the Dataquest community? I do!  I have benefited a lot from various people sharing their insights on my work. As I've progressed, I've started giving back and showing other people what I would have done differently in their notebooks. I've even started writing a generic post about the most important comments on our projects. This led me to the idea of extracting all the feedback data and gathering it in one dataset.  

I have divided this project into three stages, all of them are not that complicated on their own. But as we combine them together, it starts to look interesting:

* Part 1: Scrape the data - we'll use the Beautiful Soup library to gather all the necessary string values from the website and store them in a pandas dataframe.
* Part 2: Clean and analyse the data - we should be well accustomed to this part. Webscraping very often delivers 'dirty' text values. It is normal for the scraper to pick up a few extra signs or lines of HTML during the process. We'll have to deal with that.
* Part 3: Use machine learning models on the data. Why perform the analysis yourself, when you can send the machine to do that work for you?

Let's get to work:
# Part 1 - scraping the data from Dataquest's community forum
If you haven't used BeautifulSoup yet, then I encourage you to check my introduction notebook. It follows a similar path that we're going to take: scraping not one, but many websites. Let's have a look at how the actual Guided project post looks, so we can have a better idea on what we want to achieve:

<img width="942" alt="main" src="https://user-images.githubusercontent.com/87883118/144956101-27b15dc3-4ad2-473f-870a-faa241819d02.png">

This is the main thread of Guided Projects. It contains all of our Guided Projects, that we've decided to publish. Most of them received a reply with some comments - we're interested in the contents of that reply:

<img width="918" alt="feedback" src="https://user-images.githubusercontent.com/87883118/144956671-3c8dc0bc-1922-4a12-9c73-5e4b549d93af.png">


In this post, Michael published his project and Elena replied with some remarks to his work. We're interested in scraping only the content of Elena's remarks. It is not going to be as easy as scraping one website, because we want to scrape a specific part of many websites, to which we don't have the links...yet. Here's the plan of attack:
1. We don't have the links to all of the Guided project posts - we need to obtain them, which means we'll have to scrape the main thread of Guided Projects
2. After scraping the main thread we'll create a dataframe containing posts, titles, links and... number of replies
  * We'll filter out posts with no replies
  * The remaining dataset should contain only the posts that received feedback and the links to those posts - we can commence scraping the actual individual posts

## Step 1:

We'll begin with inspecting the contents of the whole website: https://community.dataquest.io/c/share/guided-project/55
We can use our browser for that, I personally use Chrome. Just hover your mouse above the title of the post right-click it and choose Inspect, (BUT pay attention! 
I've choosen a post that's a few posts below the top - just in case the first posts has a different class)
<!-- 
<img width="1132" alt="right_click" src="https://user-images.githubusercontent.com/87883118/144968155-70f5aee1-092d-4cda-bfa2-3c9162c6345c.png">
 -->
Now we can actually look at the code of the website, when you hover your mouse cursor above certain elements of the code in the right window, the browser will highlight that element in the left window, in the below example my cursor is hovering above the \<tr data-topic-id=...> :

<img width="1092" alt="inspect" src="https://user-images.githubusercontent.com/87883118/145134328-abb52874-0bc5-4bc9-a952-662d44fe00d6.png">

You can notice that the actual link has a class 'title raw-link raw-topic-link', it's in the second line of the code below:

```html
<a href="/t/predicting-bike-rentals-with-machine-learning-optimization-of-the-linear-model/558618/3" role="heading"
   level="2" class="title raw-link raw-topic-link" data-topic-id="558618"><span dir="ltr">Predicting Bike Rentals 
  With Machine Learning, Optimization of the Linear Model</span></a>
```
For a warm up let's try scraping all the links with that class into one list and see how many we've managed to extract:

```python
# imports:
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
# step 1 lets scrape the guided project website with all the posts:
url = "https://community.dataquest.io/c/share/guided-project/55"
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
list_all = soup.find_all("tr")
# check how many elements we've extracted:
len(list_all)
```
\[Output]: 30

Our list has only 30 elements! We were expecting a bigger number, what happened? Unfortunately we're trying to scrape a dynamic website. (a more [in depth article](https://www.zesty.io/mindshare/marketing-technology/dynamic-vs-static-websites/) on the matter). Dataquest loads only the first 30 posts when our browser opens the forums page, if we want to see more we have to scroll down. But how do we program our scraper to scroll down? [Selenium](https://selenium-python.readthedocs.io/) is a go to solution for that issue but we're going to use something much simpler: 
* scroll down to the bottom of the website
* when we reach the end save the website as file
* instead of processing a website with BeautifulSoup, we'll process that file

Let's get to scrolling down:

![20211207_132125](https://user-images.githubusercontent.com/87883118/145162982-e907978a-5ff0-49ca-8830-f377618ddb52.jpg)

Yes that is an actual fork pushing down the 'down arrow' on the keyboard, weighted down with an empty coffe cup (the author of this post does not encourage any unordinary use of cutlery or dishware around your electronic equipment). Having scrolled down to the very bottom, we can save the website using > File > Save Page As... No we can load that file into our notebook and commence scraping:
```python
import codecs
# this is the file of the website, after scrolling all the way down:
file = codecs.open("../input/dq-projects/projects.html", "r", "utf-8")
parser = BeautifulSoup(file, 'html.parser')
list_all = parser.find_all('tr')
series_4_df = pd.Series(list_all)
# create a dataframe with values(title, link, etc.) extracted from the html file:
df = pd.DataFrame(series_4_df, columns=['content'])
df['content'] = df['content'].astype(str)
df.head()
```
## We've arrived at step 2
We have created a dataframe filled with a lot of HTML code. Let's inspect the content of one cell:
```python
df.loc[2,'content'] 
```
<img width="817" alt="content_dirty" src="https://user-images.githubusercontent.com/87883118/145167073-e75870fa-4c21-4134-9849-617ffe1db9c6.png">
