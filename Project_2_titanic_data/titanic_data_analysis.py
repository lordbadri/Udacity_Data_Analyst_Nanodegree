# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:07:14 2017

@author: Badrinath
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")


titanic_df = pd.read_csv("titanic_data.csv")

#lets count the survived passengers based on sex

survived_df = pd.pivot_table(data=titanic_df, index='Sex', values='PassengerId', columns='Survived', aggfunc='count')

survived_df_1 = titanic_df[titanic_df['Survived'] == 1]
didnt_survive_df = titanic_df[titanic_df['Survived'] == 0]

# plot to see how many were males and females and how many survived

plt.figure()
p1 = sns.factorplot(x="Sex", y="Survived", data=titanic_df, kind="bar")
p1.set_ylabels("survival probability")

plt.figure()
# Create a list of colors (from iWantHue)
colors_1 = ["r","g"]
colors_2 = ["r","g"]

# Create a pie chart
plt.pie(survived_df.loc['female'],labels=['Females didnt Survive','Females Survived'],shadow=False,colors=colors_1,explode=(0, 0.15),startangle=90,autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()

plt.figure()
plt.pie(survived_df.loc['male'],labels=['Males didnt Survive','Males Survived'],shadow=False,colors=colors_2,explode=(0, 0.15),startangle=90,autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()


fig = plt.figure()

colors = ["r","g", "y"]

ax1 = fig.add_subplot(331)
ax1.pie(titanic_df.groupby("Pclass").count()["Survived"],labels=['First Class','Second Class', 'Third Class'],shadow=False,colors=colors, startangle=90,autopct='%1.1f%%')
ax1.set_title("Passengers by class")
plt.axis('equal')
plt.tight_layout()

ax2 = fig.add_subplot(332)
ax2.pie(survived_df_1.groupby("Pclass").count()["Survived"],labels=['First Class','Second Class', 'Third Class'],shadow=False,colors=colors, startangle=90,autopct='%1.1f%%')
ax2.set_title("Passengers survived by class")
plt.axis('equal')
plt.tight_layout()

ax3 = fig.add_subplot(333)
ax3.pie(didnt_survive_df.groupby("Pclass").count()['Survived'],labels=['First Class','Second Class', 'Third Class'],shadow=False,colors=colors, startangle=90,autopct='%1.1f%%')
ax3.set_title("Passengers didn't survive by class")
plt.axis('equal')
plt.tight_layout()
plt.show()