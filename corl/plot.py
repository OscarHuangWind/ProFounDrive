# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:53:23 2024

@author: Admin
"""

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
# rs = np.random.RandomState(33)
# d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                  columns=list(ascii_letters[26:]))

data = pd.DataFrame(data=np.array([[99.02, 78.84,67.53],
                                   [97.33, 95.44, 78.88],
                                   [94.49, 92.94, 97.69]]),
                    columns=['Urban', 'Highway', 'Rural'],
                    index=['Urban', 'Highway', 'Rural'])

# Compute the correlation matrix
# corr = data.corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
hfont = {'fontname': "Times New Roman"}
font_size = 24
# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data, cmap=cmap, cbar=False, square=True, linewidths=.5, annot=True, fmt=".1f", annot_kws={"size": 20})
ax.set_xlabel("Evaluation", fontsize=font_size, **hfont)
ax.set_ylabel("Sequential Training (Urban to Rural)", fontsize=font_size, **hfont)
plt.xticks(fontsize=20, **hfont)
# plt.yticks(fontsize=20)
pos, textvals = plt.yticks()
plt.yticks(pos, ('Urban', 'Highway', 'Rural'), rotation=90, fontsize=20, va="center", **hfont)
# sns.set(font_scale=2)

data = pd.DataFrame(data=np.array([[99.15, 79.02,71.17],
                                   [70.79, 99.40, 81.44],
                                   [72.55, 86.21, 99.63]]),
                    columns=['Urban', 'Highway', 'Rural'],
                    index=['Urban', 'Highway', 'Rural'])

# Compute the correlation matrix
# corr = data.corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
hfont = {'fontname': "Times New Roman"}
font_size = 24
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data, cmap=cmap, cbar=False, square=True, linewidths=.5, annot=True, fmt=".1f", annot_kws={"size": 20})
ax.set_xlabel("Evaluation", fontsize=font_size, **hfont)
ax.set_ylabel("Disjoint Training", fontsize=font_size, **hfont)
plt.xticks(fontsize=20, **hfont)
# plt.yticks(fontsize=20)
pos, textvals = plt.yticks()
plt.yticks(pos, ('Urban', 'Highway', 'Rural'), rotation=90, fontsize=20, va="center", **hfont)
# sns.set(font_scale=2)

plt.show()
