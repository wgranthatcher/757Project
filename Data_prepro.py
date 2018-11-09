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
#cover = pd.read_csv('W:/Documents/SCHOOL/Towson/2018-2022 -- DSc - Computer Security/6_Fall 2018/COSC 757 - Data Mining/Assignments/Final Project - 12-6/%s' % file, delimiter="\t" , error_bad_lines=False)

#path = "W:/Documents/SCHOOL/Towson/2018-2022 -- DSc - Computer Security/6_Fall 2018/COSC 757 - Data Mining/Assignments/Final Project - 12-6/DataPro/%s" % file
path = "/home/grant309/757Project/DataPro/%s" % file
print(path)

cover = pd.read_csv('/home/grant309/757Project/Data/amazon_reviews_us_%s' % file, delimiter="\t" , error_bad_lines=False)
#cover = pd.read_csv('/home/grant309/757Project/%s' % file, delimiter="\t", error_bad_lines=False)

folder = file

prepro = pd.DataFrame()

'''
try:
    os.makedirs(path)
except OSError:
    print("Could not create path: %s" % path)
else:
    print("Path %s created" % path)
'''


print ('Original Length: ')
print(len(cover))

#for index, row in cover.iterrows():
#for index, row in cover.itertuples():
    #print(row)
    #if len(row)==15:
    #    prepro = prepro.append(row)

cover = cover.dropna()
print(len(cover))

print(cover['star_rating'].dtypes)

cover['star_rating'] = pd.to_numeric(cover['star_rating'], errors='coerce')
cover = cover.dropna()

#print(cover['star_rating'].dtype)

prepro1 = cover.loc[lambda cover: cover['star_rating'] == 1]
#prepro.append(prepro1)
print(prepro1)

prepro2 = cover.loc[lambda cover: cover['star_rating'] == 2]
#prepro.append(prepro1)

prepro3 = cover.loc[lambda cover: cover['star_rating'] == 3]
#prepro.append(prepro1)

prepro4 = cover.loc[lambda cover: cover['star_rating'] == 4]
#prepro.append(prepro1)

prepro5 = cover.loc[lambda cover: cover['star_rating'] == 5]
#prepro.append(prepro1)

#prepro = prepro1 + prepro2 + prepro3 + prepro4 + prepro5

frames = [prepro1, prepro2, prepro3, prepro4, prepro5]
prepro = pd.concat(frames)

#print('PREPRO')
#print(prepro)
#) or (lambda cover: cover['star_rating'] == 2) (lambda cover: cover['star_rating'] == 3) or (lambda cover: cover['star_rating'] == 4), (lambda cover: cover['star_rating'] == 5)]


print('Final Length: ')
print(len(prepro))

prepro.to_csv(path, sep='\t', index=False)


	
