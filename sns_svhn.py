import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

content = np.array(['FGSM', 'PGD', 'CW '])

svhn_adv = np.array([[0.6497, 0.5266, 0.4839]])
svhn_inst = np.array([[0.4909, 0.4566, 0.3811]])
svhn_causal = np.array([[0.8784, 0.8848, 0.8994]])
svhn_treat = np.array([[0.913, 0.9141, 0.9101]])

# svhn_adv = np.array([[0.704, 0.5547, 0.5177]])
# svhn_inst = np.array([[0.4369, 0.403, 0.436]])
# svhn_causal = np.array([[0.9186, 0.9225, 0.9122]])
# svhn_treat = np.array([[0.9304, 0.9344, 0.9242]])

svhn = np.concatenate([svhn_adv, svhn_inst, svhn_causal, svhn_treat])

svhn_df = pd.DataFrame(columns=content, data=svhn)

svhn_ = svhn_df.set_index([["Adv", "CF", "CC", "AC"]])
svhn_df_ = svhn_.stack().reset_index()
svhn_df_.columns = ['', 'Method', 'Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

fig = plt.figure(figsize=(5, 3.7), dpi=600)
b = sns.barplot(x='Accuracy', y='Method', hue='', data=svhn_df_, alpha=0.9, palette="Greens_d", orient='h')

b.legend(loc='upper right', title='', frameon=True, fontsize=7)

b.set(xlim=(0, 1.13))
b.tick_params(labelsize=13)
b.set_yticklabels(labels=b.get_yticklabels(), rotation=90, va='center')

#b.axes.set_title("VGG-16 Network",fontsize=20)
b.set(ylabel=None)
b.set_xlabel("Acc", fontsize=10, loc='right')

plt.setp(b.get_legend().get_texts(), fontsize=13)
plt.tight_layout()
plt.savefig("./causal_svhn_vgg.png")

