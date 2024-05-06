import numpy as np
import pandas as pd
df = pd.read_csv('data/mydata.csv')
from sentence_transformers import SentenceTransformer
from newsextractor import getArticle

EMBED_MODEL = SentenceTransformer('sentence-transformers/sentence-t5-base', cache_folder='./cache')

articles = []

for i in df['links']:
	try:
		article = getArticle(i)
		if len(article.strip().split()) > 100:
			articles.append(article)
	except Exception as e:
		pass

emb = EMBED_MODEL.encode(articles, show_progress_bar=True)
np.save('data/mydata_t5base.npy', emb)