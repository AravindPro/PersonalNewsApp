# Get the html pages from a set of urls
# Present them in a format like: Title, summary, allow QnA
# Get 2-3 similar news from other sources
# Based on my likings filter out the news I don't like
# Allow converting webpages into rss feeds
# Save the news you found out now into json. With another read=False parameter. Create HTML based viewer
import re
import requests as req
import feedparser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FAV = ["https://www.thehindu.com/news/national/feeder/default.rss", 
       "https://www.thehindu.com/news/international/feeder/default.rss",
       "https://www.thehindu.com/opinion/editorial/feeder/default.rss",
	   ]
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')
# T5SUMMARYPIPE = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-summarize-news")
TOKENIZER = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
MODEL = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

# Load saved.json file
import json

with open('results/saved.json') as f:
	DATA = json.load(f)

class NewsContent:
	title = ""
	summary = ""
	link = ""
	embedding = ""
	contenttext = ""
	def __init__(self, title, summary, link, content):
		self.title = prettifytext(title)
		self.summary = prettifytext(summary)
		self.link = link
		self.contenttext = prettifytext(content)
		# self.embedding = list(EMBEDDING_MODEL.encode(summary))

def getInfos(url):
	NewsFeed = feedparser.parse(url)
	news = []

	for i in NewsFeed.entries:
		for j in DATA['news']:
			# If already present in the saved.json file, then re compute the summary, etc
			# Use the same details
			if(i.link == j['link']):
				news.append(NewsContent(j['title'], j['summary'], j['link'], j['contenttext']))
				continue
		try:
			soup = BeautifulSoup(req.get(i.link).text, 'html.parser')
			text = soup.get_text()

			soup = BeautifulSoup(i.summary, 'html.parser')
			if(soup.find()):
				news.append(NewsContent(i.title, soup.get_text(), i.link, text))
			else:
				news.append(NewsContent(i.title, i.summary, i.link, text))

		except Exception as e:
			print(e)
	return news

def getSummary(text):
	inputs = TOKENIZER.encode(text, return_tensors='pt')
	output = MODEL.generate(inputs, num_beams=2, max_length=300, early_stopping=True)
	return TOKENIZER.batch_decode(output)
	
def prettifytext(text):
	text = text.strip()
	text = re.sub('\n+', '\n', text)
	text = text.replace('\t', ' ')
	# Replace multiple spaces with single space
	text = re.sub(r'\s+', ' ', text)
	return text
if __name__=="__main__":
	url = "https://www.thehindu.com/news/national/feeder/default.rss"
	news = getInfos(url)
	
	# Save news as JSON in saved.json
	
	newdata = {'news': []}
	for i in news:
		newdata['news'].append(i.__dict__)

	try:
		with open('saved.json', 'w') as f:
			json.dump(newdata, f)
	except Exception as e:
		print(e)
		with open('saved.json', 'w') as f:
			json.dump(DATA, f)