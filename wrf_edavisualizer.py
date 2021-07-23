# -*- coding: utf-8 -*-
"""
Spyder Editor

this script opens the raw stiv grib files and extracts the midwest region
into hourly precip files in the ../stivnpys folder
"""
import streamlit as st
from sklearn import datasets
import sklearn
from netCDF4 import Dataset
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
import pandas as pd

os.chdir('/media/ats/Backup/data/wrfexample')
#open example file and get listing of variable names
ncf=Dataset('/media/ats/Backup/data/cleveland/wrfout_d01_2019-05-26_06:15:00')
print(ncf.variables.keys())
lats=ncf.variables['XLAT'][0,:,:]
lons=ncf.variables['XLONG'][0,:,:]
#%%

#get a listing of all files
b=os.listdir('/media/ats/Backup/data/cleveland')
#listing of variables we want to extract and set aside in seperate files
#for now lets look at single level sfc variables
vs=['Q2','T2','U10','V10','PBLH','HFX','QFX','RAINNC']

for v in vs:
    outf=np.zeros((241,36,36))
    for ix,x in enumerate(b):
        c=Dataset('/media/ats/Backup/data/cleveland/'+x)
        outf[ix,:,:]=c.variables[v][0,:,:]
    np.save(v+'.npy',outf)
    print(v)

#%%
vs=['HFX','QFX','T2']
pts=[[18,18],[5,5]]


vf=np.load(vs[0]+'.npy')

for iv,v in enumerate(vs):
    vf=np.load(v+'.npy')
    for ipt,pt in enumerate(pts):
        if ipt==0:
            df = pd.DataFrame(vf[:,pt[0],pt[1]])
            df['Points'] = str(pt[0])+'_'+str(pt[1])
            df.columns=[v,'Points']
        else:
            df0 = pd.DataFrame(vf[:,pt[0],pt[1]])
            df0['Points'] = str(pt[0])+'_'+str(pt[1])
            df0.columns=[v,'Points']
            df=pd.concat([df,df0])
            print(df)
    if iv==0:
        df.to_csv('0.csv')
    else:
        dfa=pd.read_csv('0.csv',index_col=0)
        dfa[v]=df[v]
        dfa.to_csv('0.csv')
        
sns.jointplot(data=dfa, x='HFX', y='QFX', hue="Points")

