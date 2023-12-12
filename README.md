# PersonalNewsApp
A news app that collects news from the sources you like via RSS feed or via a webpage by inspecting a certain element. It has integrated AI for summarizing the news for you in a tree-like fashion (allows you to delve deep into the part of the news that interests you). 
Also, an AI for filtering out the news that isn't in your interests. 

# Aim 
- Don't rely on a single news source.
- Set the tone/emotion of the article to suit your needs (I prefer a neutral tone). Don't rely on the writer's opinions; build your own. Facts are the important things.
- If you don't understand something (say, a certain topic mentioned in the article that isn't described in detail), you are free to ask questions and get your facts right.
- FOCUS MODE: There is a lot of news and everything doesn't matter to you. Find only the things that you are interested in.
- EXPLORE MODE: Find out new topics that might interest you.
- STORY TELLING MODE: Your time is valuable. So, let AI summarize the news and feed it to you in "bits" as a story. 
  
# The flow of the app
- You open it to get the news for you in a language that generates excitement. 
- Then, the news is categorized into 5-7 topics (magic number), each having its highlights/summary in a paragraph in a neutral tone.
- You can open the news headlines to view them along with a summary.
- You can then select a news headline and do one of the following: ask any questions regarding the news or enter storytelling mode, which describes the whole news for you.
- The program automatically gets news from many more websites on the same topic (then filters out non-relevant articles, stores information pieces, and removes copies). 

# TODO
[x] RSS reader, extract embeddings, summarize article using T5.
[x] Store title, summary, content as JSON.
[ ] Store the content along with embeddings. 
