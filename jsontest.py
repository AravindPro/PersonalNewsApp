import json
with open('saved.json') as f:
	DATA = json.load(f)

print(DATA['news'])