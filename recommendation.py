from pathlib import Path
from bs4 import BeautifulSoup
import faiss
import numpy as np
import pandas as pd
import requests
# from main import getArticle
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer(
	'sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')


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
if __name__=="__main__":
	df = pd.read_csv(f"{Path(__file__).parent.absolute()}/data/mydata.csv")
	vectors = []

	count = 1
	for i in df['links']:
		print(i)
		v = EMBEDDING_MODEL.encode(getArticle(i), convert_to_numpy=True)
		vectors.append(v)

		if count%100 == 0:
			np.save(f'{Path(__file__).parent.absolute()}/data/vectors.h5', np.array(vectors))
	np.save(f'{Path(__file__).parent.absolute()}/data/vectors.h5', np.array(vectors))
	# print(getArticle('https://www.business-standard.com/economy/news/93-of-2-000-rupee-notes-worth-rs-3-32-trillion-returned-since-may-rbi-123090100772_1.html'))

