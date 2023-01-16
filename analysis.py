import pandas as pd
import matplotlib.pyplot as plt

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