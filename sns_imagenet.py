import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

content = np.array(['VGG-FGSM', 'VGG-PGD', 'VGG-CW', 'Res-FGSM', 'Res-PGD', 'Res-CW'])

imgn_adv = np.array([[0.3894, 0.378, 0.4251, 0.3504, 0.3384, 0.3869]])
imgn_inst = np.array([[0.3603, 0.3517, 0.1487, 0.2466, 0.2368, 0.1664]])
imgn_causal = np.array([[0.5363, 0.5445, 0.5338, 0.5216, 0.519, 0.5071]])
imgn_treat = np.array([[0.5688, 0.5705, 0.5521, 0.5777, 0.5805, 0.5235]])

imgn = np.concatenate([imgn_adv, imgn_inst, imgn_causal, imgn_treat])

imgn_df = pd.DataFrame(columns=content, data=imgn)

imgn_ = imgn_df.set_index([["Adv", "Inst", "Causal", "Treat"]])
imgn_df_ = imgn_.stack().reset_index()
imgn_df_.columns = ['', 'Method', 'Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

fig = plt.figure(figsize=(6, 9), dpi=600)
b = sns.barplot(x='Accuracy', y='Method', hue='', data=imgn_df_, alpha=0.9, palette="Reds_d", orient='h')

b.legend(loc='upper right', title='', frameon=True)

b.set(xlim=(0, 0.8))
b.tick_params(labelsize=13)
b.set_yticklabels(labels=b.get_yticklabels(), rotation=90, va='center')

#b.axes.set_title("VGG-16 Network",fontsize=20)
b.set_ylabel("Method", fontsize=15)
b.set_xlabel("Accuracy", fontsize=15)

plt.setp(b.get_legend().get_texts(), fontsize=13)
plt.tight_layout()
plt.savefig("./causal_imgn.png")

