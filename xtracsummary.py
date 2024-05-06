# %%
import numpy as np
from recommendation import getArticle
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer(
	'sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')

def rerankedSentences(text):
	para = [i.strip() for i in text.split('\n')]
	A = EMBEDDING_MODEL.encode(para)
	q = EMBEDDING_MODEL.encode([text,]).reshape(-1)
	R = A@q
	r = np.argsort(-R)
	for i in r:
		print(para[i])
		print('-'*10)

# %%
if __name__=="__main__":
	article = getArticle(
		"https://en.wikipedia.org/wiki/Bill_Gates")

	rerankedSentences(article)
# %%
