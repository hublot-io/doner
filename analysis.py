from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from collections import Counter

dataset = load_dataset("data/fish-label", split="train")
dataset_length = len(dataset)
print("Dataset length: ", dataset_length)
print(dataset[0])
gt_parses = [d['ground_truth'] for d in dataset]
gt_parses = [json.loads(d)['gt_parse']['ner'] for d in gt_parses]
dataframe = pd.DataFrame(data=gt_parses)
print(dataframe)

keys = set()
datas = {}
for line in dataframe:
    print(line)
    try: 
        tags = [d['tag'] for d in dataframe[line]]
        for t in tags:
            keys.add(t)
    except:
        pass
for key in keys:
    datas[key] = []
for line in dataframe:
    try: 
        for d in dataframe[line]:
            datas[d['tag']].append(d['word'])
    except:
        pass

end_datas = {}
for key in keys:
    col = datas[key]
    count = Counter(col)
    end_datas[key] = count

print(end_datas)

for key in keys:
    fig, ax = plt.subplots()
    tags = end_datas[key]
    names = list(tags.keys())
    values = list(tags.values())
    ax.bar(names, values, label=names)
    plt.savefig('analysis/'+key+  '.jpg')

## gen weights: 
test_dataset = load_dataset("data/fish-label", split="train")
weights = []
for sample in test_dataset:
    ners = json.loads(sample['ground_truth'])
    ners = ners['gt_parse']['ner']
    maxs = []
    for item in ners:
        tag = item['tag']
        word = item['word']
        try:
            for i in range(0,3):
                is_max = True if word == end_datas[tag].most_common()[i][0] else False
                maxs.append(is_max)
        except:
            maxs.append(False)
    score =  sum(maxs) / len(maxs) 
    print("LineScore: ", score) 
    