import json
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
ner = pd.read_csv("ner.csv")
grouped = ner.groupby("Class")
plot = grouped.plot(kind="bar",x="Label", y=["Count"],subplots=True )
classes = set(ner["Class"])
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]

for (i,fig) in enumerate(figs):
    fig.set_figheight(8)
    fig.set_figwidth(16)
    plt.tight_layout()
    fig.savefig( "analysis/{}.png".format(i),bbox_inches='tight')

# once ploted lets find the weights of each classes



# weights = []

# for group in grouped:
#     print("g",group)

#     total = 0
#     for entry in group:
#         print(entry)
#         total += entry.label
#     print("total for group", total)
# for sample in grouped:
#     ners = json.loads(sample['ground_truth'])
#     ners = ners['gt_parse']['ner']
#     maxs = []
#     for item in ners:
#         tag = item['tag']
#         word = item['word']
#         try:
#             for i in range(0,3):
#                 is_max = True if word == end_datas[tag].most_common()[i][0] else False
#                 maxs.append(is_max)
#         except:
#             maxs.append(False)
#     score =  sum(maxs) / len(maxs) 
#     weights.append(score)
# return weights        
