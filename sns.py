import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset = np.array(['Adv', 'Inst', 'Causal', 'Treat'])

cifar10_vgg = np.array([[0.4474, 0.2538, 0.7452, 0.7831]])
svhn_vgg = np.array([[0.5266, 0.3177,  0.8645, 0.9071]])
imagenet_vgg = np.array([[0.378, 0.1704, 0.5465, 0.575]])

vgg = np.concatenate([cifar10_vgg, svhn_vgg, imagenet_vgg])
vgg_df = pd.DataFrame(columns=dataset, data=vgg)
vgg_df = vgg_df.set_index([["CIFAR10", "SVHN", "ImageNet"]])

vgg_df_ = vgg_df.stack().reset_index()
vgg_df_.columns = ['', 'Dataset', 'Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

fig = plt.figure(figsize=(10, 4), dpi=600)
b = sns.barplot(x='Dataset', y='Accuracy', hue='', data=vgg_df_, alpha=0.9, palette="Blues_d")
b.legend(loc='upper left', title='', frameon=False)
b.set(ylim=(0, 1.05))
b.tick_params(labelsize=13)
#b.axes.set_title("VGG-16 Network",fontsize=20)
b.set_xlabel("Dataset", fontsize=15)
b.set_ylabel("Accuracy", fontsize=15)

plt.setp(b.get_legend().get_texts(), fontsize=13)
plt.tight_layout()
plt.savefig("./causal_vgg.png")

#VGG 16
dataset = np.array(['CIFAR-10', 'SVHN', 'ImageNet'])

adv_vgg = np.array([[0.4474, 0.5266, 0.3780]])
inst_vgg = np.array([[0.2538, 0.3177, 0.1704]])
causal_vgg = np.array([[0.7452, 0.8645, 0.5465]])
treat_vgg = np.array([[0.7831, 0.9071, 0.575]])

vgg = np.concatenate([adv_vgg, inst_vgg, causal_vgg, treat_vgg])
vgg_df = pd.DataFrame(columns=dataset, data=vgg)
vgg_df = vgg_df.set_index([["Adv Acc", "Inst Acc", "Causal Acc", "Treat Acc"]])

vgg_df_ = vgg_df.stack().reset_index()
vgg_df_.columns = ['', 'Dataset', 'Accuracy']

matplotlib.rc_file_defaults()
sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

fig = plt.figure(figsize=(6, 4), dpi=600)
b = sns.barplot(x='Dataset', y='Accuracy', hue='', data=vgg_df_, alpha=0.9, palette="Blues_d")
b.legend(loc='upper right', title='', frameon=True)
b.set(ylim=(0, 1.00))
b.tick_params(labelsize=13)
#b.axes.set_title("VGG-16 Network",fontsize=20)
b.set(xlabel=None)
b.set_ylabel("Accuracy", fontsize=15)

plt.setp(b.get_legend().get_texts(), fontsize=13)
plt.tight_layout()
plt.savefig("./causal_vgg.png")





