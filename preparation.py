# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:01:41 2022

@author: PRANJ
"""

from nilearn import datasets
from nilearn import plotting
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
abide = datasets.fetch_abide_pcp(data_dir=r"C:\Users\PRANJ\ML_fmri_ABIDE\data",
                                #n_subjects=50,
                                pipeline="cpac",
                                quality_checked=True,
                                )
print(len(abide.func_preproc))
parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')
atlas_filename = parcellations.scale064
plotting.plot_roi(atlas_filename, draw_cross=False)
fmri_filenames = abide.func_preproc[0]
print(fmri_filenames)
first_Img = image.index_img(fmri_filenames, 1)
plotting.plot_stat_map(first_Img, threshold = 'auto')
masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                           standardize=True, 
                           memory='nilearn_cache', 
                           verbose=1)

time_series = masker.fit_transform(fmri_filenames)
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
print(correlation_matrix.shape)
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8),   
                     labels=range(time_series.shape[-1]),
                     vmax=0.8, vmin=-0.8, reorder=False)
# make list of filenames
fmri_filenames = abide.func_preproc
# load atlas
multiscale = datasets.fetch_atlas_basc_multiscale_2015()
atlas_filename = multiscale.scale064
# initialize masker object
masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                           standardize=True, 
                           memory='nilearn_cache', 
                           verbose=0)
# initialize correlation measure
correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True,
                                         discard_diagonal=True)

all_features = [] # here is where we will put the data (a container)
'''
for i,sub in enumerate(fmri_filenames):
    # extract the timeseries from the ROIs in the atlas
    time_series = masker.fit_transform(sub)
    # create a region x region correlation matrix
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # add to our container
    all_features.append(correlation_matrix)
    # keep track of status
    print('finished %s of %s'%(i+1,len(fmri_filenames)))
'''    
feat_file = r'C:\Users\PRANJ\ML_fmri_ABIDE\output\ABIDE_BASC064_features.npz'
X_features = np.load(feat_file)['a']

print(X_features.shape)
phenotypic = pd.read_csv(r"C:\Users\PRANJ\ML_fmri_ABIDE\Phenotypic_V1_0b_preprocessed1.csv")

file_ids = []
# get the file IDs from the file names
for f in fmri_filenames:
    file_ids.append(f[-27:-20])

y_asd = []
for i in range(len(phenotypic)):
    for j in range(len(file_ids)):
        if file_ids[j] in phenotypic.FILE_ID[i]:
            y_asd.append(phenotypic.DX_GROUP[i])

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_features, # x
                                                  y_asd, # y
                                                  test_size = 0.4, # 60%/40% split  
                                                  shuffle = True, # shuffle dataset before splitting
                                                  stratify = y_asd,  # keep distribution of ASD consistent between sets
                                                  random_state = 123 # same split each time
                                                 )

sns.countplot(x = y_train, label = "train")
sns.countplot(x = y_val, label = "test")
l_svc = LinearSVC(max_iter=100000) # more iterations than the default
l_svc.fit(X_train, y_train)

# predict
y_pred_svc = cross_val_predict(l_svc, X_train, y_train, cv=10)
# scores
acc_svc = cross_val_score(l_svc, X_train, y_train, cv=10)

print("Accuracy:", acc_svc)
print("Mean accuracy:", acc_svc.mean())
'''
reg_log = LogisticRegression()
#reg_log.fit(X_train, y_train)
y_pred_log = cross_val_predict(reg_log, X_train, y_train, cv=10)

acc_svc = cross_val_score(reg_log, X_train, y_train, cv=10)

print("Accuracy:", acc_svc)
print("Mean accuracy:", acc_svc.mean())
'''

gnb = GaussianNB()
gnb.fit(X_train, y_train, sample_weight=None)

# predict
y_pred_gnb = cross_val_predict(gnb, X_train, y_train, cv=10)
# scores
acc_gnb = cross_val_score(gnb, X_train, y_train, cv=10)

print("Accuracy:", acc_gnb)
print("Mean accuracy:", acc_gnb.mean())


reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, y_train)
y_pred_rf = cross_val_predict(reg_rf, X_train, y_train, cv=10)
# scores
acc_rf = cross_val_score(reg_rf, X_train, y_train, cv=10)

print("Accuracy:", acc_rf)
print("Mean accuracy:", acc_rf.mean())

reg_knn = KNeighborsClassifier()
reg_knn.fit(X_train, y_train)
y_pred_knn = cross_val_predict(reg_knn, X_train, y_train, cv=10)
acc_knn = cross_val_score(reg_knn, X_train, y_train, cv=10)
print("Accuracy:", acc_knn)
print("Mean accuracy:", acc_knn.mean())



