# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:09:18 2017

@author: Badrinath
"""

import pandas as pd
from astroML.plotting import hist
import matplotlib.pyplot as plt
#import numpy as np

df_stroop = pd.read_excel(r"D:\google_drive\Online courses\udacity\Nano_degree_data_analytics\P1\stroop.xlsx") 
df_stroop["Difference"] = df_stroop["Congruent"] - df_stroop["Incongruent"]
print (df_stroop.mean())
print (df_stroop.std())
print (df_stroop.describe())


plt.figure()
plt.hist(df_stroop["Congruent"], bins=8, normed = False,color='blue')
plt.axis([0,40,0,10])
plt.title("Histogram of the congruent words condition")
plt.xlabel("Time in seconds")
plt.ylabel("Counts")

plt.figure()
plt.hist(df_stroop["Incongruent"], bins=10, normed = False,color='red')
plt.axis([0,40,0,10])
plt.title("Histogram of the incongruent words condition")
plt.xlabel("Time in seconds")
plt.ylabel("Counts")

plt.figure()
hist(df_stroop["Congruent"], bins="knuth", normed = False,histtype='stepfilled',alpha=0.2,color='blue', label='Congruent condition')
hist(df_stroop["Incongruent"], bins="knuth", normed = False,histtype='stepfilled',alpha=0.4,color='red', label='Incongruent condition')
plt.axis([0,40,0,13])
plt.title("Histogram of the congruent and incongruent words condition")
plt.xlabel("Time in seconds")
plt.ylabel("Counts")
plt.ylabel("Counts")
plt.legend(loc='upper left')


df_pure = df_stroop
del df_pure["Difference"]

color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
df_pure.plot.box(color=color, sym='r+')
plt.ylabel('Time in seconds')
plt.show()