''' IMPORTING ALL THE LIBRARIES'''

import matplotlib.pyplot as plt
import seaborn as sns
import preprocess

import os
print(os.getcwd())

plt.figure(figsize=(10, 6))
sns.histplot(preprocess.data_set["Phrase_Length"], bins=30, kde=True)
sns.histplot(preprocess.data_frm["Phrase_Length"], bins=30, kde=True,color='purple')
plt.title("Distribution of Phrase Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Number of Phrases")

plt.show()
