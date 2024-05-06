# Get the html pages from a set of urls
# Present them in a format like: Title, summary, allow QnA
# Get 2-3 similar news from other sources
# Based on my likings filter out the news I don't like
# Allow converting webpages into rss feeds
# Save the news you found out now into json. With another read=False parameter. Create HTML based viewer
# %%
import os
import pickle
import re
from pathlib import Path
from time import sleep
import numpy as np
import requests as req
import feedparser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from gpt4free import you
import faiss
# from functools import cache
from cachetools import cached
from urllib.parse import urlparse
from newsextractor import *


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

# %%


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

def getNewFeed(lastcheck=1000, limit=10000):
	# Fetchs and returns the Dataframe of the new content
	global HISTORY
	historylinks = set(HISTORY['links'][:lastcheck])
	# print(historylinks)
	newlinks = []

	count = 1
	for i in range(len(RSS['links'])):
		print(RSS['links'][i])
		
		for j in getFeed(RSS['links'][i]):
			if count == limit:
				break
			if j['links'] not in historylinks:
				newlinks.append({"links":j['links'], "title": j['title'], "status": 0, "category": RSS['category'][i]})
				count += 1

				
	df = pd.DataFrame(newlinks)
	HISTORY = pd.concat([HISTORY, df], ignore_index=True)
	HISTORY.to_csv(f'{Path(__file__).parent.absolute()}/data/history.csv')
	return df

def addToHistory(df):
	global HISTORY
	HISTORY = pd.concat([HISTORY, df], ignore_index=True)
	HISTORY.to_csv(f'{Path(__file__).parent.absolute()}/data/history.csv')

def printNews(feed: pd.DataFrame):
	for i in range(len(feed)):
		print(f"{i}. {feed['title'][i]}")

	inp = ""
	while inp != 'q':
		inp = input("Enter the article number")
		try:
			if inp[0] == 'a':
				print(getArticle(feed['links'][int(inp[1:])]))
			else:
				print(getSummaryYou(getArticle(feed['links'][int(inp)])))
		except Exception as e:
			print(e)
		print("---"*50)

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

def getSimilarityHistory(emb, k):
	his = np.load('./data/mydata_allmini.npy')
	index = faiss.IndexFlatIP(384)
	index.add(his)

	D, I = index.search(emb, k=k)
	D=D.mean(axis=1)
	return D

def querySimilarity(emb, q):
	q = EMBEDDING_MODEL.encode([q,]).reshape(-1)
	return emb@q

def getMoodRanking(df, k, mood):
	articles = []
	for i in df['links']:
		articles.append(getArticle(i))
	# D = getSimilarityHistory(articles, k=k).mean(axis=1)
	# if q is not None:
	# 	D += querySimilarity(articles, q)
		
	# Filter based on mood
	D = querySimilarity(articles, mood)
	return D
def getHistoryScores(articles, k):
	D = getSimilarityHistory(articles, k=k).mean(axis=1)
	return D

def addToQueue(newfeed: pd.DataFrame, scores: np.ndarray):
	concated = pd.concat([newfeed, pd.DataFrame(scores, columns=["score"])], axis=1)
	pd.concat([QUEUE, concated])

def saveintempfile(articlesdf, M, H):
	df = pd.DataFrame(columns=['title', 'links', 'h_score', "m_score"])
	df['title'] = articlesdf['title']
	df['links'] = articlesdf['links']
	df['h_score'] = H
	df['m_score'] = M

	df.to_csv('filtered.csv')
	

def faissSimilarity(articles, q):
	import faiss
	arrarticles = EMBEDDING_MODEL.encode(articles)
	q = EMBEDDING_MODEL.encode([q,]).reshape(-1)

	vector_dimension = arrarticles.shape[1]
	index = faiss.IndexFlatL2(vector_dimension)
	index.add(arrarticles)
	k = index.ntotal
	distances, ann = index.search(q, k=k)
	return distances
	
def getFeedEmbedding(df):
	articles = []
	for i in df['links']:
		articles.append(getArticle(i))
	return EMBEDDING_MODEL.encode(articles, show_progress_bar=True)

if __name__=="__main__":
	if not os.path.exists(f'{Path(__file__).parent.absolute()}/data/history.csv'):
		pd.DataFrame(columns=['links', 'title', 'status', 'category', 'summary']).to_csv(
			f'{Path(__file__).parent.absolute()}/data/history.csv')
	if not os.path.exists(f'{Path(__file__).parent.absolute()}/data/rss.csv'):
		pd.DataFrame(columns=['links', 'category']).to_csv(
			f'{Path(__file__).parent.absolute()}/data/rss.csv')

	# cache_file_path = 'my_cache.cache'

	# # Create a file-based cache
	# try:
	# 	with open('mycache.pickle', 'rb') as f:
	# 		cache = pickle.load(f)
	# except FileNotFoundError:
	# 	cache = {}

	HISTORY = pd.read_csv(
		f'{Path(__file__).parent.absolute()}/data/history.csv', index_col=0)
	print("History loaded...")

	RSS = pd.read_csv(
		f'{Path(__file__).parent.absolute()}/data/sources.csv', index_col=0)
	print("Sources loaded...")
	# FAV = ["https://www.thehindu.com/news/national/feeder/default.rss",
	#        "https://www.thehindu.com/news/international/feeder/default.rss",
	#        "https://www.thehindu.com/opinion/editorial/feeder/default.rss",
	# 	   ]
	EMBEDDING_MODEL = SentenceTransformer(
		'sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')
	print("Loaded embedding model...")

	if (os.path.exists(f'{Path(__file__).parent.absolute()}/data/queue.json')):
		QUEUE = pd.read_json(
			f'{Path(__file__).parent.absolute()}/data/queue.json')
	else:
		QUEUE = pd.DataFrame(
			columns=["links", "title", "status", "category", "scores"])

	# Save a file for: new feed having news with text and another file for visited links 
	# Save a file for embeddings.
	# It must generate a simple 

	# Get a dict of history links
	# addNewElementsToRss(['https://www.thehindu.com/news/national/feeder/default.rss','https://cdn.technologyreview.com/topnews.rss'])
	# with open('data/feed.txt', 'w') as f:
	# 	f.write(prevSummary())
	# print(getSummaryYou(
	# 	f"Generate summary of: \n{getArticle('https://www.technologyreview.com/2024/01/12/1086442/the-innovation-that-gets-an-alzheimers-drug-through-the-blood-brain-barrier/')}"))

	df = getNewFeed()
	print(df.head())
	FEEDEMB = getFeedEmbedding(df)

	HISTHRESH = 0.5
	MOODTHRESH = 0.1

	mood = "Science"
	if mood is not None:
		M = querySimilarity(FEEDEMB, q="Tech")

	H = getSimilarityHistory(FEEDEMB, k=3)
	print(H.shape)
	# saveintempfile(df, M=M, H=H)
	# df.columns.append("scores")
	df["scores"] = H
	print(df.head())
	df = df[(M>MOODTHRESH)&(H>HISTHRESH)]
	QUEUE = pd.concat([QUEUE, df], ignore_index=True)
	QUEUE = QUEUE.sort_values("scores", ascending=False, ignore_index=True)
	print(QUEUE.head())
	QUEUE.to_json('data/queue.json')
	# df = pd.read_csv('filtered.csv')
	# D = getFeedRanking(df, k=3, mood="Tech")
	# print(D.shape)
	# print(D)
	# print(D)
	# print(D.shape)

	# Dind = np.max(D, axis=1)

	# sortedindex = np.argsort(-D)

	# addToQueue(df.iloc[sortedindex[:10]], D[sortedindex[:10]])
	# print(QUEUE.head())
	# for i in sortedindex[:10]:
	# 	# if D[i] < 0.6:
	# 	# 	break
	# 	print(D[i])
	# 	print(df['title'][i])
	# 	print(df['links'][i])
	# addToHistory(df.iloc[sortedindex[:10]])

	# print(getFeed('https://indianexpress.com/section/india/'))
	# print(getArticle('https://openai.com/research/dall-e-3-system-card'))
