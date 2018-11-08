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

for index, row in cover.iterrows():
    #print(row)
    if len(row)==15:
        prepro = prepro.append(row)

print('Final Length: ')
print(prepro)

prepro.to_csv(path, sep='\t', index=False)


	
