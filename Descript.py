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


#file = raw_input("Input File Name: ")
#color = raw_input("Input Hist Color: ")

#Read Auto-MPG dataset into a Pandas data frame (mde .tsv  "Tab Seperated Value")
#cover = pd.read_csv('/home/grant309/757Project/Data/amazon_reviews_us_Apparel_v1_00.tsv', delimiter="\t" , error_bad_lines=False)

data = ['Apparel_v1_00',
'Automotive_v1_00',
'Baby_v1_00',
'Beauty_v1_00',
'Books_v1_00',
#'Books_v1_01',
#'Books_v1_02',
'Camera_v1_00'#,
#'Digital_Ebook_Purchase_v1_00',
#'Digital_Ebook_Purchase_v1_01',
#'Digital_Music_Purchase_v1_00',
#'Digital_Software_v1_00',
#'Digital_Video_Download_v1_00',
#'Digital_Video_Games_v1_00',
#'Electronics_v1_00',
#'Furniture_v1_00',
#'Gift_Card_v1_00',
#'Grocery_v1_00',
#'Health_Personal_Care_v1_00',
#'Home_Entertainment_v1_00',
#'Home_Improvement_v1_00',
#'Home_v1_00',
#'Jewelry_v1_00',
#'Kitchen_v1_00',
#'Lawn_and_Garden_v1_00',
#'Luggage_v1_00',
#'Major_Appliances_v1_00',
#'Mobile_Apps_v1_00',
#'Mobile_Electronics_v1_00',
#'Musical_Instruments_v1_00',
#'Music_v1_00',
#'Office_Products_v1_00',
#'Outdoors_v1_00',
#'PC_v1_00',
#'Personal_Care_Appliances_v1_00',
#'Pet_Products_v1_00',
#'Shoes_v1_00',
#'Software_v1_00',
#'Sports_v1_00',
#'Tools_v1_00',
#'Toys_v1_00',
#'Video_DVD_v1_00',
#'Video_Games_v1_00',
#'Video_v1_00',
#'Watches_v1_00',
#'Wireless_v1_00'
]

#path = "/home/grant309/757Project/Figures/%s" % file
#print(path)

text_file = open('/home/grant309/757Project/Results.txt', "w+")
text_file.write("RESULTS\n")
text_file.close()

for cats in data:
    #cover = pd.read_csv('/home/grant309/757Prject/Data/amazon_reviews_us_%s.tsv' % file, delimiter="\t" , error_bad_lines=False)
    cover = pd.read_csv('/home/grant309/757Project/DataPro/%s.tsv' % cats, delimiter="\t", error_bad_lines=False)
    
    text_file = open('/home/grant309/757Project/Results.txt', "a")
    text_file.write(cats + "\n")
    text_file.write(str(cover.describe()) + "\n")
    
    zero_votes = cover.loc[lambda cover: cover['helpful_votes'] != 0]
    one_votes = cover.loc[lambda cover: cover['helpful_votes'] != 1]
    two_votes = cover.loc[lambda cover: cover['helpful_votes'] != 2]
    
    text_file.write("Helpfule Votes:  0 = " + str(len(zero_votes)) + " 1 = " + str(len(one_votes)) + " 2 = " + str(len(two_votes)) + "\n")
    
    zero_votes = cover.loc[lambda cover: cover['total_votes'] != 0]
    one_votes = cover.loc[lambda cover: cover['total_votes'] != 1]
    two_votes = cover.loc[lambda cover: cover['total_votes'] != 2]
    
    text_file.write("Helpfule Votes:  0 = " + str(len(zero_votes)) + " 1 = " + str(len(one_votes)) + " 2 = " + str(len(two_votes)) + "\n")
    text_file.write("\n")
    
    print(cats + " Done")
    
    text_file.close()

'''	
try:
    os.makedirs(path)
except OSError:
    print("Could not create path: %s" % path)
else:
    print("Path %s created" % path)
'''

