# Get the html pages from a set of urls
# Present them in a format like: Title, summary, allow QnA
# Get 2-3 similar news from other sources
# Based on my likings filter out the news I don't like
# Allow converting webpages into rss feeds
# Save the news you found out now into json. With another read=False parameter. Create HTML based viewer
import os
import re
from pathlib import Path
from time import sleep
import requests as req
import feedparser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from gpt4free import you

if not os.path.exists(f'{Path(__file__).parent.absolute()}/data/history.csv'):
	pd.DataFrame(columns=['links', 'title', 'status', 'category', 'summary']).to_csv(
		f'{Path(__file__).parent.absolute()}/data/history.csv')
if not os.path.exists(f'{Path(__file__).parent.absolute()}/data/rss.csv'):
	pd.DataFrame(columns=['links', 'category']).to_csv(
		f'{Path(__file__).parent.absolute()}/data/rss.csv')
	

HISTORY = pd.read_csv(f'{Path(__file__).parent.absolute()}/data/history.csv', index_col=0)
RSS = pd.read_csv(f'{Path(__file__).parent.absolute()}/data/rss.csv', index_col=0)

# FAV = ["https://www.thehindu.com/news/national/feeder/default.rss", 
#        "https://www.thehindu.com/news/international/feeder/default.rss",
#        "https://www.thehindu.com/opinion/editorial/feeder/default.rss",
# 	   ]
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')
# T5SUMMARYPIPE = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-summarize-news")
# TOKENIZER = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
# MODEL = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

VISITEDLINKS = {}
# Load saved.json file
import json

# class NewsContent:
# 	title = ""
# 	summary = ""
# 	link = ""
# 	embedding = ""
# 	contenttext = ""
# 	def __init__(self, title, summary, link, content):
# 		self.title = prettifytext(title)
# 		self.summary = prettifytext(summary)
# 		self.link = link
# 		self.contenttext = prettifytext(content)
# 		# self.embedding = list(EMBEDDING_MODEL.encode(summary))

# def getInfos(url):
# 	NewsFeed = feedparser.parse(url)
# 	for i in NewsFeed.entries:
# 		for j in DATA['news']:
# 			# If already present in the saved.json file, then re compute the summary, etc
# 			# Use the same details
# 			if(i.link == j['link']):
# 				news.append(NewsContent(j['title'], j['summary'], j['link'], j['contenttext']))
# 				continue
# 		try:
# 			soup = BeautifulSoup(req.get(i.link).text, 'html.parser')
# 			text = soup.get_text()

# 			soup = BeautifulSoup(i.summary, 'html.parser')
# 			if(soup.find()):
# 				news.append(NewsContent(i.title, soup.get_text(), i.link, text))
# 			else:
# 				news.append(NewsContent(i.title, i.summary, i.link, text))

# 		except Exception as e:
# 			print(e)
# 	return news

# def getSummary(text):
# 	inputs = TOKENIZER.encode(text, return_tensors='pt')
# 	output = MODEL.generate(inputs, num_beams=2, max_length=300, early_stopping=True)
# 	return TOKENIZER.batch_decode(output)
	
def prettifytext(text):
	text = text.strip()
	text = re.sub('\n+', '\n', text)
	text = text.replace('\t', ' ')
	# Replace multiple spaces with single space
	text = re.sub(r'\s+', ' ', text)
	return text

def addNewElementsToRss(array, category="Default"):
	global RSS
	setlinks = set(RSS['links'])
	newelements = []

	for i in array:
		if i not in setlinks:
			newelements.append({"links": i, "category": category})

	RSS = pd.concat([RSS, pd.DataFrame(newelements)], ignore_index=True)
	RSS.to_csv(f'{Path(__file__).parent.absolute()}/data/rss.csv')
	print('Added successfully')

def fetchNewFeed(lastcheck=100):
	global HISTORY
	historylinks = set(HISTORY['links'][:lastcheck])
	newlinks = []

	summary = ""
	count = 1
	for i in range(len(RSS['links'])):
		print(RSS['links'][i])
		feed = feedparser.parse(RSS['links'][i])
		for j in feed['entries']:
			if j['link'] not in historylinks:
				s = getSummaryYou(getArticle(j['link']))
				newlinks.append({"links":j['link'], "title": j['title'], "status": 0, "category": RSS['category'][i], "summary": s})

				summary += j['title'] + '\n'
				summary += s+'-'*40+'\n'

				if count%20 == 0:
					print("Saving...")
					df = pd.DataFrame(newlinks)
					HISTORY = pd.concat([HISTORY, df], ignore_index=True)
					HISTORY.to_csv(f'{Path(__file__).parent.absolute()}/data/history.csv')
				count += 1
	df = pd.DataFrame(newlinks)
	HISTORY = pd.concat([HISTORY, df], ignore_index=True)
	HISTORY.to_csv(f'{Path(__file__).parent.absolute()}/data/history.csv')
	return summary

def fetchNewFeedPrint(lastcheck=100):
	global HISTORY
	historylinks = set(HISTORY['links'][:lastcheck])
	newlinks = []

	for i in range(len(RSS['links'])):
		print(RSS['links'][i])
		feed = feedparser.parse(RSS['links'][i])
		for j in feed['entries']:
			if j['link'] not in historylinks:
				newlinks.append({"links":j['link'], "title": j['title'], "status": 0, "category": RSS['category'][i], "summary": ""})


	for i in range(len(newlinks)):
		print(f"{i}. {newlinks[i]['title']}")

	inp = ""
	while inp != 'q':
		inp = input("Enter the article number")
		try:
			if inp[0]=='a':
				print(getArticle(newlinks[int(inp[1:])]['links']))
			else:
				print(getSummaryYou(getArticle(newlinks[int(inp)]['links'])))
		except Exception as e:
			print(e)
		print("---"*50)
	df = pd.DataFrame(newlinks)
	HISTORY = pd.concat([HISTORY, df], ignore_index=True)
	HISTORY.to_csv(f'{Path(__file__).parent.absolute()}/data/history.csv')

def getnochilderennodes(tag):
	nochildrennodes = []
	for i in tag.children:
		try:
			if (len(list(i.children)) != 0):
				nochildrennodes.extend(getnochilderennodes(i))
		except:
			if (len(i.text.split()) > 20):
				if (i.name == None):
					nochildrennodes.append(i.parent)
				elif (i.name == 'p' or i.name == 'div' or i.name == 'span'):
					nochildrennodes.append(i)
	return nochildrennodes


def getArticle(url):
	soup = BeautifulSoup(req.get(url).text, 'html.parser')
	ps = soup.find_all('p')
	tot = 0
	for i in ps:
		tot += len(i.text.split())
	if (tot < len(soup.html.text.split())/2):
		leafelements = getnochilderennodes(soup.html)
	else:
		leafelements = ps

	# Take only parents of meaningful leaf nodes
	childcount = {}
	for i in leafelements:
		if (i.name == 'div' or i.name == 'p' or i.name == 'span'):
			if (i.parent not in childcount):
				childcount[i.parent] = 1
			else:
				childcount[i.parent] += 1
	text = ""

	avg = sum([len(i.text.split()) for i in childcount])/len(childcount)
	print(f"Average: {avg}")
	for i in childcount:
		if (childcount[i] >= 1 and len(i.text.split()) > min(avg, 200)):
			text += i.text
			text += '\n'
		# print(i.text)
		# print(len(i.text.split()))
		# # print(childcount[i])
		# print('--'*50)
	return text
def getSummaryYou(text):
	ans = you.Completion.create(f"Generate summary of: \n{text}")
	if ans.text != "Unable to fetch the response, Please try again.":
		return ans.text
	else:
		sleep(2)
		return getSummaryYou(text)

def prettifytext(text):
	text = text.strip()
	text = re.sub('\n+', '\n', text)
	text = text.replace('\t', ' ')
	# Replace multiple spaces with single space
	text = re.sub(r'\s+', ' ', text)
	return text
def prevSummary(prev=10):
	summary = ""
	count = 1
	for i in range(len(HISTORY['summary'])):
		summary += f"{HISTORY['title'][i]}\n\n{HISTORY['summary'][i]}\n{'--'*20}\n"
		count += 1
		if(count == prev):
			break
	return summary
if __name__=="__main__":
	# Save a file for: new feed having news with text and another file for visited links 
	# Save a file for embeddings.
	# It must generate a simple 

	# Get a dict of history links
	addNewElementsToRss(['https://www.thehindu.com/news/national/feeder/default.rss','https://cdn.technologyreview.com/topnews.rss'])
	# with open('data/feed.txt', 'w') as f:
	# 	f.write(prevSummary())
	# print(getSummaryYou(
	# 	f"Generate summary of: \n{getArticle('https://www.technologyreview.com/2024/01/12/1086442/the-innovation-that-gets-an-alzheimers-drug-through-the-blood-brain-barrier/')}"))
	fetchNewFeedPrint()