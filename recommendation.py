# %%
from pathlib import Path
import re
from bs4 import BeautifulSoup
import faiss
import numpy as np
import pandas as pd
import requests
# from main import getArticle
from scipy.spatial.distance import cdist, squareform
from sentence_transformers import SentenceTransformer
from trafilatura import fetch_url, extract

EMBEDDING_MODEL = SentenceTransformer(
	'sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')
# %%

def getnochilderennodes(tag):
	nochildrennodes = []
	for i in tag.children:
		try:
			if (len(list(i.children)) != 0):
				nochildrennodes.extend(getnochilderennodes(i))
		except:
			if (len(i.text.split()) > 10):
				if (i.name == None):
					nochildrennodes.append(i.parent)
				elif (i.name == 'p' or i.name == 'div' or i.name == 'span'):
					nochildrennodes.append(i)
	return nochildrennodes


def getArticle(url):
	try:
		soup = BeautifulSoup(requests.get(url).text, 'html.parser')
		ps = soup.find_all('p')
		tot = 0
		for i in ps:
			tot += len(i.text.split())
		if (tot <= len(soup.html.text.split())/2):
			leafelements = getnochilderennodes(soup.html)
		else:
			leafelements = ps

		if(len(leafelements) == 0):
			return soup.html.text
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
	except Exception as e:
		return ""

def getArticleTrafiluatura(url):
	downloaded = fetch_url(url)
	res = extract(downloaded)
	print(res)
	return res
def fromcsv(csv_path):
	df = pd.read_csv(csv_path)
	vectors = []

	count = 1
	for i in df['links']:
		print(i)
		article = getArticleTrafiluatura(i)
		if article is not None:
			v = EMBEDDING_MODEL.encode(article, convert_to_numpy=True)
			vectors.append(v)

			if count%30 == 0:
				print("Saving...")
				np.save(f'{Path(__file__).parent.absolute()}/data/mydata', np.array(vectors))
			count+=1
	np.save(f'{Path(__file__).parent.absolute()}/data/mydata', np.array(vectors))
	# print(getArticle('https://www.business-standard.com/economy/news/93-of-2-000-rupee-notes-worth-rs-3-32-trillion-returned-since-may-rbi-123090100772_1.html'))
def getSimilarity(articles, k):
	arr = np.load('./data/mydata.npy')
	index = faiss.IndexFlatIP(384)
	index.add(arr)

	D, I = index.search(EMBEDDING_MODEL.encode(articles), k=k)
	return D

def getSimilarityBetween(a1, a2, measure="cosine"):
	# index = faiss.IndexFlatIP(384)
	# index.add(arr1)
	# D, I = index.search(arr2, k=4)
	arr1 = EMBEDDING_MODEL.encode(a1)
	arr2 = EMBEDDING_MODEL.encode(a2)
	if measure=="cosine":
		return arr1@arr2.T
	elif measure=="euclidean":
		# a1.extend(a2)
		# a1 = EMBEDDING_MODEL.encode(a1)
		# a2 = EMBEDDING_MODEL.encode(a2)
		return cdist(arr1, arr2, metric='euclidean')
		# distance_matrix = squareform(distances)
		# np.fill_diagonal(distance_matrix, np.inf)
		# return distance_matrix
	
def splitarticletosentences(article):
	pattern = r'[.?!]|[\n]'
	split = re.split(pattern, article)
	split = [item.strip() for item in split if not all(char.isspace() for char in item)]
	return split
def numview(arrsents):
	for i in range(len(arrsents)):
		print(f"{i}. {arrsents[i]}")
def comparearticles(url1, url2, threshold=0.6, measure="cosine", greater=True):
	a1 = splitarticletosentences(getArticle(url1))
	a2 = splitarticletosentences(getArticle(url2))
	numview(a1)
	print("--"*20)
	numview(a2)

	D = getSimilarityBetween(a1, a2, measure=measure)

	if greater:
		Dg = D>threshold
	else:
		Dg = D<threshold
	for i in range(len(Dg)):
		for j in range(len(Dg[0])):
			if Dg[i, j]:
				print(a1[i])
				print(a2[j])
				print("--"*10)

# %%
if __name__ == "__main__":
	# arr = np.load('./data/mydata.npy')
	# index = faiss.IndexFlatIP(384)
	# index.add(arr)

	# D, I = index.search(EMBEDDING_MODEL.encode([
	# 	getArticle("https://techcrunch.com/2024/01/19/openai-signs-up-its-first-higher-education-customer-arizona-state/"),

	# ]), k=4)
	# 	# "Vans, Supreme owner VF Corp says hackers stole 35 million customers personal data",
	# 	# "OpenAI signs up its first higher education customer, Arizona State",
	# 	# "Apple offers EU set of pledges aimed at settling Apple Pay antitrust probe",
	# 	# "X is rolling out audio and video calls to Android",
	# 	# "General Catalyst eyes VC deal in India push"
	# 	# getArticle("https://techcrunch.com/2024/01/19/apple-pay-eu-commitments/"),
	# 	# getArticle("https://timesofindia.indiatimes.com/hot-picks/top-benefits-of-tea-tree-oil-for-your-skin-hair-and-more/articleshow/106986941.cms"),
	# 	# getArticle("https://timesofindia.indiatimes.com/auto/cars/2024-hyundai-creta-rivalling-kia-seltos-diesel-mt-launched-at-12-lakh-details/articleshow/106978207.cms")
	# 		# getArticle("https://interestingengineering.com/military/drone-in-a-box-on-wheels"),
	# 		# getArticle("https://interestingengineering.com/reviews/7-factors-to-consider-when-buying-an-air-purifier"),
	# 		# getArticle("https://interestingengineering.com/innovation/ai-death-calculator-lifespan-income-personality")
	# 	# getArticle(
	# 	# 	"https://www.teslarati.com/elon-musk-response-rimac-nevera-23-records-in-one-day/")
	# 	# ]), 4
	# 	# 			 )
	# # index.
	# print(D)

	# Combine the patterns into a single regular expression using the '|' (or) operator
	# pattern = r'[.?!]|[\n]'
	# a1 = getArticle("https://www.theverge.com/2024/1/19/24044319/openai-chip-manufacturing-fundraising")
	# a2 = getArticle(
	# 	"https://www.thehindu.com/sci-tech/technology/sam-altman-seeks-raise-billions-network-ai-chip-factories-report/article67758682.ece")
	# a1 = splitarticletosentences(a1)
	# a2 = splitarticletosentences(a2)
	# print(a1)
	# print('--'*20)
	# print(a2)
	# D = getSimilarityBetween(a1,a2)
	# comparearticles(
	# 	"https://www.gadgets360.com/wearables/news/apple-vision-pro-preorders-price-storage-variants-specifications-repair-cost-4897329",
	# 	"https://www.indiatvnews.com/technology/news/apple-vision-pro-now-open-for-pre-orders-512gb-and-1tb-options-available-2024-01-20-912806",
	# 	measure="euclidean",
	# 	threshold=0.9,
	# 	greater=False
	# )
	fromcsv("./data/mydata.csv")

# %%
