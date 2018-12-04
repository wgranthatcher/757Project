from __future__ import division
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

data = [
'Apparel_v1_00'#,
#'Automotive_v1_00',
#'Baby_v1_00',
#'Beauty_v1_00',
#'Books_v1_00',
#
#'Books_v1_01',
#'Books_v1_02',
#
#'Camera_v1_00',
#
#'Digital_Ebook_Purchase_v1_00',
#'Digital_Ebook_Purchase_v1_01',
#'Digital_Music_Purchase_v1_00',
#
#'Digital_Software_v1_00'#,
#
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

text_file = open('/home/grant309/757Project/BasicResults.txt', "w+")
text_file.write("BASIC ML RESULTS\n")
text_file.close()

#pd.set_option('float_format', '{:f}'.format)

for cats in data:
    cover = pd.read_csv('/home/grant309/757Project/DataPro/%s.tsv' % cats, delimiter="\t", error_bad_lines=False)
    
    lenc = LabelEncoder()
    cover['vine'] = lenc.fit_transform(cover['vine'])
    cover['verified_purchase'] = lenc.fit_transform(cover['verified_purchase'])
    cover['product_id'] = lenc.fit_trnasform(cover['product_id'])

    #print(cover[['vine','verified_purchase']])
    #cover.replace(['Y','N'],['1','0'])
    #print(cover[['vine','verified_purchase']])

    text_file = open('/home/grant309/757Project/BasicResults.txt', "a+")
    text_file.write(cats + "\n")
    text_file.write(str(cover.describe()) + "\n")
    
    train=cover.sample(frac=0.7,random_state=1234)
    test=cover.drop(train.index)

    print(train[['vine','verified_purchase']])
    
    base = [#'customer_id',
    'helpful_votes',
    'product_id',
    'product_parent',
    #'product_title',
    #'review_body',
    #'review_date',
    #'review_headline',
    #'review_id',
    'star_rating',
    'total_votes',
    'verified_purchase',
    'vine'
    ]

    for b in base:
        
        print("Obs_Bin: ")
        
        obs_bin = ['customer_id',
        'helpful_votes',
        'product_id',
        'product_parent',
        #'product_title',
        #'review_body',
        #'review_date',
        #'review_headline',
        #'review_id',
        'star_rating',
        'total_votes',
        'verified_purchase',
        'vine'
        ]
        
        print(obs_bin)
        print(obs_bin.index(b))
        
        print("B: ")
        print(b)
        
        obs_bin.remove(str(b))
        print("Obs_Bin")
        print(obs_bin)

        labs = cover[b]
        labs = list(set(labs))
        print("labs:")
        print(labs)
        
        cls = [str(b)]
        #print(list(cls))
        trainObs = train[obs_bin]
        trainObs = trainObs.values
        #print(list(trainObs))
        trainCls = train[str(b)]
        trainCls = trainCls.values.ravel()
        #print(list(testObs))
        testObs = test[obs_bin]
        testObs = testObs.values
        #print(list(trainCls))
        testCls = test[str(b)]
        testCls = testCls.values.ravel()
        #print(list(testCls))

        # ----  K Nearest Neighbor Classification
        text_file.write("---- KNN ----")
        text_file.write("\n")
        knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
        knn.fit(trainObs, trainCls)
        knn_pred = knn.predict(testObs)
        text_file.write(str(knn_pred))
        text_file.write("\n")
        text_file.write("KNN Accuracy:")
        text_file.write("\n")
        text_file.write(str((sum(testCls==knn_pred))/len(knn_pred)))
        text_file.write("\n")
        knn_tab = confusion_matrix(testCls, knn_pred, labels=labs)
        text_file.write(str(knn_tab))
        text_file.write("\n")
        text_file.write(str(metrics.classification_report(testCls, knn_pred)))
        text_file.write("\n")
        
        # ---- Decision Tree Classification
        text_file.write("---- Decision Tree ----")
        text_file.write("\n")
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(trainObs, trainCls)
        dt_pred = clf.predict(testObs)
        text_file.write(str(dt_pred))
        text_file.write("\n")
        text_file.write("DT Accuracy:")
        text_file.write("\n")
        text_file.write(str((sum(testCls==dt_pred))/len(dt_pred)))
        text_file.write("\n")
        dt_tab = confusion_matrix(testCls, dt_pred, labels=labs)
        text_file.write(str(dt_tab))
        text_file.write("\n")
        text_file.write(str(metrics.classification_report(testCls, dt_pred)))
        text_file.write("\n")
        
        # ---- Decision Tree Entropy Classification
        text_file.write("---- Decision Tree Entropy ----")
        text_file.write("\n")
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(trainObs, trainCls)
        text_file.write(str(clf))
        text_file.write("\n")
        dt_pred = clf.predict(testObs)
        text_file.write(str(dt_pred))
        text_file.write("\n")
        text_file.write("DT Entropy Accuracy:")
        text_file.write("\n")
        text_file.write(str((sum(testCls==dt_pred))/len(dt_pred)))
        text_file.write("\n")
        dt_tab = confusion_matrix(testCls, dt_pred, labels=labs)
        text_file.write(str(dt_tab))
        text_file.write("\n")
        text_file.write(str(metrics.classification_report(testCls, dt_pred)))
        text_file.write("\n")
        
        # ---- Random Forest Classifier
        text_file.write("---- Random Forest ----")
        text_file.write("\n")
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(trainObs, trainCls)
        text_file.write(str(clf))
        text_file.write("\n")
        rf_pred = clf.predict(testObs)
        text_file.write(str(rf_pred))
        text_file.write("\n")
        text_file.write("RF Accuracy:")
        text_file.write("\n")
        text_file.write(str((sum(testCls==rf_pred))/len(rf_pred)))
        text_file.write("\n")
        rf_tab = confusion_matrix(testCls, rf_pred, labels=labs)
        text_file.write(str(rf_tab))
        text_file.write("\n")
        text_file.write(str(metrics.classification_report(testCls, rf_pred)))
        text_file.write("\n")
        
        # ---- Naive Bayes Classifier
        text_file.write("---- Naive Bayes ----")
        text_file.write("\n")
        gnb = GaussianNB()
        gnb = gnb.fit(trainObs, trainCls)
        text_file.write(str(gnb))
        text_file.write("\n")
        nb_pred = gnb.predict(testObs)
        text_file.write(str(nb_pred))
        text_file.write("\n")
        text_file.write("NB Accuracy:")
        text_file.write("\n")
        text_file.write(str((sum(testCls==nb_pred))/len(nb_pred)))
        text_file.write("\n")
        nb_tab = confusion_matrix(testCls, nb_pred, labels=labs)
        text_file.write(str(nb_tab))
        text_file.write("\n")
        text_file.write(str(metrics.classification_report(testCls, nb_pred)))
        text_file.write("\n")
        


    print(cats + " Done")
    
    text_file.close()



