import re
from bs4 import BeautifulSoup
import requests as req

# Given a list of elements and frequency either return the head
# element that has its children as the corresponding news articles
# Criterion for being a head element is it has >= 5 articles under it.	
# Ignore the "Load more option for now"
def getheadelements(tagcounts: dict):
	values = tagcounts.values()
	heads = []
	for i in tagcounts:
		if(tagcounts[i] > 4):
			heads.append()
	
	if(len(heads) == 0):
		pass

# def extractdivfornews(url):
# 	soup = BeautifulSoup(req.get(url).text)
# 	ps = soup.find_all('a')

# def getContent(url):
# 	soup = BeautifulSoup(req.get(url).text, 'html.parser')
# 	ps = soup.find_all('p')
	
# 	tot = 0
# 	for i in ps:
# 		tot += len(i.text.split())
# 	# if()
# 	parents = {}
# 	for i in ps:
# 		if(i.parent not in parents):
# 			parents[i.parent] = 1
# 		else:
# 			parents[i.parent] += 1

# 	return parents



	# return parents
if __name__ == "__main__":
	url = 'https://www.kdnuggets.com/are-we-undervaluing-simple-models'
	# elements = getContent(url)
	# for i in elements:
	# 	# if(len(i.text) > 500):
	# 	print(i.text)
	# 	print(elements[i])
	# 	print('--'*10)
	# for i in getnochilderennodes(url):
	# 	print(i.text)
	# 	print('--'*50)
	with open('test.txt', 'w', errors='ignore') as f:
		f.write(prettifytext(getArticle(url)))

# Criterion to use:
# no of children tags within must be more than 2
# no of words in parent must be more than 400
# Length of text in leaf (if not p) must be more than 20 (50 for longer articles)