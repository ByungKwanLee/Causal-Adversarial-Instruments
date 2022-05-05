import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

content = np.array(['FGSM', 'PGD', 'CW '])

# cifar_adv = np.array([[0.4984, 0.4474, 0.4322]])
# cifar_inst = np.array([[0.2418, 0.2447, 0.2447]])
# cifar_causal = np.array([[0.7789, 0.7738, 0.7188]])
# cifar_treat = np.array([[0.7939, 0.7965, 0.7845]])

cifar_adv = np.array([[0.5215, 0.475, 0.4554]])
cifar_inst = np.array([[0.1574, 0.1609, 0.1755]])
cifar_causal = np.array([[0.807, 0.7976, 0.7975]])
cifar_treat = np.array([[0.838, 0.8431, 0.8148]])

cifar = np.concatenate([cifar_adv, cifar_inst, cifar_causal, cifar_treat])

cifar_df = pd.DataFrame(columns=content, data=cifar)

cifar_ = cifar_df.set_index([["Adv", "CF", "CC", "AC"]])
cifar_df_ = cifar_.stack().reset_index()
cifar_df_.columns = ['', 'Method', 'Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

fig = plt.figure(figsize=(5, 3.7), dpi=600)
b = sns.barplot(x='Accuracy', y='Method', hue='', data=cifar_df_, alpha=0.9, palette="Blues_d", orient='h')

b.legend(loc='upper right', title='', frameon=True, fontsize=7)

b.set(xlim=(0, 1.05))
b.tick_params(labelsize=13)
b.set_yticklabels(labels=b.get_yticklabels(), rotation=90, va='center')

#b.axes.set_title("VGG-16 Network",fontsize=20)
#b.set_ylabel("Method", fontsize=15)
b.set(ylabel=None)
b.set_xlabel("Acc", fontsize=10, loc='right')

plt.setp(b.get_legend().get_texts(), fontsize=13)
plt.tight_layout()
plt.savefig("./causal_cifar_res.png")

