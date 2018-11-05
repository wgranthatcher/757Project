import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
import seaborn as sns
matplotlib.style.use('ggplot')


file = raw_input("Input File Name: ")

#Read Auto-MPG dataset into a Pandas data frame (mde .tsv  "Tab Seperated Value")
#cover = pd.read_csv('/home/grant309/757Project/Data/amazon_reviews_us_Apparel_v1_00.tsv', delimiter="\t" , error_bad_lines=False)

path = "/home/grant309/757Project/Figures/%s" % file
print(path)

cover = pd.read_csv('/home/grant309/757Project/Data/amazon_reviews_us_%s.tsv' % file, delimiter="\t" , error_bad_lines=False)

folder = file


try:
    os.makedirs(path)
except OSError:
    print("Could not create path: %s" % path)
else:
    print("Path %s created" % path)

#Histograms

#DISTPLOT for numerical values
plt.figure(figsize=(8,6))
ax = sns.distplot(cover['customer_id'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/customer_id" % folder)

print("customer_id complete")

plt.figure(figsize=(8,6))
ax = sns.distplot(cover['product_parent'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/product_parent" % folder)

print("product_parent complete")

'''
plt.figure(figsize=(8,6))
ax = sns.distplot(cover['star_rating'].dropna(), color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("star_rating")

print("star_rating complete")
''' 

#COUNTPLOT for categoorical values

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['star_rating'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/star_rating" % folder)

print("star_rating complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['helpful_votes'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/helpful_votes" % folder)

print("helpful_votes complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['total_votes'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/total_votes" % folder)

print("total_votes complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['vine'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/vine" % folder)

print("vine complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['verified_purchase'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/verified_purchase" % folder)

print("verified_purchase complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['review_date'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/review_date" % folder)

print("review_date complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['product_id'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/product_id" % folder)

print("product_id complete")

plt.figure(figsize=(8,6))
ax = sns.countplot(cover['product_category'], color='blue')
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
#plt.show()
plt.savefig("/home/grant309/757Project/Figures/%s/product_category" % folder)

print("product_category complete")

#Each has unique Review ID
#review_id skipped

#Product Title is same as ID (unique to product)
#Skipped product_title

#Review Headline should be unique...?
#skipped review_headline

#Review Body should be unique...?
#skipped review_body


'''
#Massive Scatter Matrix!!
from pandas.plotting import scatter_matrix

scatter_matrix(cover[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Cover_Type"]],figsize = [16, 16],marker = ".", s = 0.2,diagonal="kde")
plt.show()
'''


#Traditional Plots (matplotlib)
'''
plt.figure(figsize=(8,6))
plt.hist(police.age, bins=30, color='xkcd:teal')
plt.xlabel("age",fontsize=20)
plt.ylabel("Counts",fontsize=20)
plt.title("",fontsize=20)
plt.show()
'''



#Traditional KDE Plots (matplotlib)
'''
plt.figure(figsize=(8,6))
ax = police.manner_of_death_bin.plot.kde()
ax.set_xlabel("manner_of_death",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

'''

#Booleans --> Error
'''
plt.figure(figsize=(8,6))
ax = police.body_camera.astype(int).plot.kde()
ax.set_xlabel("body_camera",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()

#booleans --> Error

plt.figure(figsize=(8,6))
ax = police.signs_of_mental_illness.astype(int).plot.kde()
ax.set_xlabel("signs_of_mental_illness",fontsize=20)
ax.set_ylabel("",fontsize=20)
ax.set_title("",fontsize=20)
plt.show()
'''


#Scatterplot (matplotlib)
'''
ax = cars2.plot.scatter('weight','mpg')
ax.set_xlabel("Weight",fontsize=20)
ax.set_ylabel("MPG",fontsize=20)
ax.set_title("Scatterplot of MPG by Weight",fontsize=20)
plt.show()
'''

