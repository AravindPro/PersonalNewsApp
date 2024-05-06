import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests as req
import feedparser
from trafilatura import fetch_url, extract

# Given a list of elements and frequency either return the head
# element that has its children as the corresponding news articles
# Criterion for being a head element is it has >= 5 articles under it.	
# Ignore the "Load more option for now"


def getArticle(url):
	downloaded = fetch_url(url)
	res = extract(downloaded)
	# print(res)
	return res

def getFeed(url):
	# Code to get a array of news feed from rss/website.
	def getFeedWebsite(url):
		def numas(ele):
			count = 0
			for i in ele.find_all('a'):
				if (i != None and len(i.text.strip().split()) > 7):
					count += 1
			return count
		soup = BeautifulSoup(req.get(url).text, 'html.parser')
		aall = soup.find_all('a')

		for i in aall:
			if (len(i.text.strip().split()) > 7):
				ai = i
				break
		p = ai.parent

		while (numas(p) < 10 and p.parent != None):
			p = p.parent
		feed = []
		for i in p.find_all('a'):
			if len(i.text.strip().split()) > 7:
				l = i['href']
				if l[0] != 'h':
					parsed_url = urlparse(url)
					l = f"{parsed_url.scheme}://{parsed_url.netloc}"+l
				feed.append({"links": l, "title": i.text.strip()})
		return feed

	response = req.get(url)
	content_type = response.headers.get("Content-Type", "")
	try:
		if 'rss' in content_type or 'xml' in content_type:
			feeder = feedparser.parse(url)
			feed = []
			for i in feeder['entries']:
				feed.append({"links": i['link'], "title": i['title']})
			return feed
		elif 'html' in content_type:
			return getFeedWebsite(url)
	except Exception as e:
		print(f"Error {e} in {url}")
		return []

	# return parents


def getArticleo(url):
	soup = BeautifulSoup(req.get(url).text, 'html.parser')
	ps = soup.find_all('p')
	totp = 0
	for i in ps:
		totp += len(i.text.split())

	leafelements = getnochilderennodes(soup.body)
	totl = 0
	for i in leafelements:
		totl += len(i.text.split())

	if (totp > len(soup.body.text.split())/2):
		leafelements = ps
	elif (totl > len(soup.body.text.split())/2):
		leafelements = leafelements
	else:
		return soup.body.text

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

# def getFeedWebsite(url):
	
if __name__ == "__main__":
	# url = 'https://www.kdnuggets.com/are-we-undervaluing-simple-models'
	# # elements = getContent(url)
	# # for i in elements:
	# # 	# if(len(i.text) > 500):
	# # 	print(i.text)
	# # 	print(elements[i])
	# # 	print('--'*10)
	# # for i in getnochilderennodes(url):
	# # 	print(i.text)
	# # 	print('--'*50)
	# with open('test.txt', 'w', errors='ignore') as f:
	# 	f.write(prettifytext(getArticle(url)))
	res = getFeed('https://openai.com/news/product')
	for i in res:
		print(i)

# Criterion to use:
# no of children tags within must be more than 2
# no of words in parent must be more than 400
# Length of text in leaf (if not p) must be more than 20 (50 for longer articles)