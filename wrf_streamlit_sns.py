# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 11:45:43 2021

@author: bob
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
from scipy.spatial import KDTree
#%%
os.chdir('/media/ats/Backup/data/wrfexample')
#open example file and get listing of variable names
ncf=Dataset('/media/ats/Backup/data/cleveland/wrfout_d01_2019-05-26_06:15:00')
print(ncf.variables.keys())
lats=ncf.variables['XLAT'][0,:,:]
lons=ncf.variables['XLONG'][0,:,:]
tree = KDTree(np.c_[lons.ravel(), lats.ravel()])

latmin,latmax,lonmin,lonmax=lats.min(),lats.max(),lons.min(),lons.max()

vlist=['Q2','T2','U10','V10','PBLH','HFX','QFX']
vlistcopy=vlist.copy()
v1 = st.sidebar.selectbox('select variable 1', tuple(vlist))
vlistcopy.remove(v1)
v2 = st.sidebar.selectbox('select variable 2', tuple(vlistcopy))
# st.sidebar.write('Point 1')
# latslider=st.sidebar.slider('Choose Lat Point 1',1,36)
# lonslider=st.sidebar.slider('Choose Lon Point 1',1,36)
# st.sidebar.write('Point 2')
# latslider2=st.sidebar.slider('Choose Lat Point 2',1,36)
# lonslider2=st.sidebar.slider('Choose Lon Point 2',1,36)


st.sidebar.write('Point 1')
latslider=st.sidebar.text_input('Choose Lat Point 1',value=39)
lonslider=st.sidebar.text_input('Choose Lon Point 1',value=-84)
st.sidebar.write('Point 2')
latslider2=st.sidebar.text_input('Choose Lat Point 2',value=40)
lonslider2=st.sidebar.text_input('Choose Lon Point 2',value=-83)

def outofbounds(inval,vmin,vmax):
    if inval>vmax or inval<vmin:
        return True
    else:
        return False


if outofbounds(float(latslider),latmin,latmax)==True or outofbounds(float(lonslider),lonmin,lonmax)==True:
    st.sidebar.write('Point 1 is out of bounds')

if outofbounds(float(latslider2),latmin,latmax)==True or outofbounds(float(lonslider2),lonmin,lonmax)==True:
    st.sidebar.write('Point 1 is out of bounds')                             


dd, ii = tree.query([[lonslider, latslider]])
nptx=int(np.floor(ii[0]/36))
npty=ii[0]-nptx*36

dd, ii = tree.query([[lonslider2, latslider2]])
npty2=int(np.floor(ii[0]/36))
nptx2=ii[0]-npty2*36

st.sidebar.write('Point1 lon {} and lat {}, point2 lon {} and lat {}'.format(nptx,npty,nptx2,npty2))  

latslider,lonslider,latslider2,lonslider2=npty+1,nptx+1,npty2+1,nptx2+1

jttype = st.sidebar.selectbox('joint plot type', ('scatter','kde','hist','hex'))

# cmap = plt.get_cmap('jet')
# c=amr[labs==(cluster_number-1)]
# cm=np.mean(c,axis=0)
# cmap = plt.get_cmap('jet')




vs=[v1,v2]
pts=[[latslider-1,lonslider-1],[latslider2-1,lonslider2-1]]


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

col1, col2 = st.beta_columns([2,2])
with col1:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig=plt.figure(figsize=(3,3),dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.set_xlim(lons.min(),lons.max()) 
    ax.set_ylim(lats.min(),lats.max())
    # a=np.where((hours==x) & (conc[:,2]==0))[0]
    #am=np.mean(aa[:,:,17:,:,:],axis=(0,1,2))
    #print(am.max())
    ax.scatter(lons[latslider-1,lonslider-1],lats[latslider-1,lonslider-1])
    ax.scatter(lons[latslider2-1,lonslider2-1],lats[latslider2-1,lonslider2-1])
    plt.show()
    st.pyplot(fig)
with col2:       
    fig1=plt.figure(figsize=(3,3),dpi=100)       
    sns.jointplot(data=dfa, x=v1, y=v2, hue="Points",kind=jttype)
    st.pyplot()
    
fig2=plt.figure(figsize=(6,3),dpi=100)  
for ipt,pt in enumerate(pts):
    sns.lineplot(data=dfa[dfa.Points==str(pt[0])+'_'+str(pt[1])][v1], color="g")
ax2 = plt.twinx()
for ipt,pt in enumerate(pts):
    sns.lineplot(data=dfa[dfa.Points==str(pt[0])+'_'+str(pt[1])][v2], color="b",ax=ax2)
#sns.lineplot(data=df.column2, color="b", ax=ax2)
st.pyplot()


# st.title("streamlit demo")

# # st.write("""
# #          # Test Header
         
# #          Test line
         
# #          """)
# col1, col2, col3 = st.beta_columns(3)
# with col1:

#     level = st.selectbox('select level', ('850 hPa', '500 hPa'))
# with col2:

#     classifier_name = st.selectbox('select classifier', ('Kmeans','Spectral'))
# with col3:

#     randornot=st.radio('Random Cluster State?', ('No', 'Yes'))

# def get_params(classifier_name,randornot):
#     params=dict()
#     if classifier_name == 'Kmeans':
#         K=st.slider('# clusters',2,9)
#         params['n_clusters'] = K
#     elif classifier_name == 'Spectral':
#         K=st.slider('# clusters',2,9)
#         params['n_clusters'] = K
#     if randornot=='Yes':
#         params['randornot']=np.random.randint(1,high=100)
#     else:
#         params['randornot']=0
#     # elif classifier_name == 'SVM':
#     #     c=st.sidebar.slider('C',1,10)
#     #     params['C'] = c
#     # elif classifier_name == 'Random Forest':
#     #     m=st.sidebar.slider('max_depth',2,15)
#     #     params['max_depth'] = m
#     #     ne=st.sidebar.slider('n_estimators',10,200)
#     #     params['n_estimators'] = ne
#     return params
# @st.cache(suppress_st_warning=True)
# def get_class(classifier_name, params):
#     if classifier_name == 'Kmeans':
#         st.write('using ', classifier_name, 'with ', params['n_clusters'], ' clusters')
#         clf=KMeans(n_clusters=params['n_clusters'],random_state=params['randornot'])
#         kmeans=clf.fit(X)
#         labs=kmeans.labels_
#         cc=np.reshape(kmeans.cluster_centers_,(p['n_clusters'],5,9))
#     elif classifier_name == 'Spectral':
#         st.write('using ', classifier_name, 'with ', params['n_clusters'], ' clusters')
#         clf=SpectralClustering(n_clusters=params['n_clusters'],random_state=params['randornot'])
#         kmeans=clf.fit(X)
#         labs=kmeans.labels_
#         cc=np.zeros((params['n_clusters'],5,9))
#         Xr=np.reshape(X,(1365,5,9))
#         for x in range(params['n_clusters']):
#             cc0=Xr[labs==x]
#             cc[x]=np.mean(cc0,axis=0)
#     # elif classifier_name == 'SVM':
#     #     clf=SVC(C=params['C'])
#     # elif classifier_name == 'Random Forest':
#     #     clf=RF(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

#     return labs,cc

# def get_dataset(level):
#     if level == '850 hPa':
#         data=hgtsr[:,0,:]
#     elif level == '500 hPa':
#         data=hgtsr[:,2,:]

#     X = data

#     return X
    
# X = get_dataset(level)
# p=get_params(classifier_name,randornot)
# labs,cc=get_class(classifier_name,p)
# #xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=125)
# # kmeans=clf.fit(X)
# # labs=kmeans.labels_
# # cc=np.reshape(kmeans.cluster_centers_,(p['n_clusters'],5,9))
# amr=np.reshape(np.mean(aa[:,:,17:,:,:],axis=(2)),(1365,90,90))
# #ypr=clf.predict(xte)

# #st.write('Number of Members in Each Cluster')
# clstmbr=[]
# clstcnt=[]
# for x in range(p['n_clusters']):
#     c=amr[labs==x]
#     #st.write('Cluster '+str(x+1)+': '+str(len(c)))
#     clstcnt.append(len(c))
#     clstmbr.append(x+1)

# #for dropdown to select cluster member
# #cluster_number = st.selectbox('select cluster member', tuple(clstmbr))
# #for slider to select cluster member
# cluster_number = st.slider('select cluster member', min(clstmbr),max(clstmbr))

# st.write('Cluster: '+str(cluster_number)+', '+str(clstcnt[cluster_number-1])+' members')

# #plot the cluster centers and mean precip for cluster center
# col1, col2 = st.beta_columns([3,2])
# with col2:
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     cmap = plt.get_cmap('jet')
#     c=amr[labs==(cluster_number-1)]
#     cm=np.mean(c,axis=0)
#     cmap = plt.get_cmap('jet')
#     fig=plt.figure(figsize=(5,5),dpi=200)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.coastlines()
#     ax.add_feature(cfeature.STATES)
#     ax.set_xlim(xpw,xpe) 
#     ax.set_ylim(yps,ypn)
#     # a=np.where((hours==x) & (conc[:,2]==0))[0]
#     #am=np.mean(aa[:,:,17:,:,:],axis=(0,1,2))
#     #print(am.max())
#     ax.pcolormesh(lonaa,lataa,cm,vmin=.1,vmax=.4,cmap=cmap)
#     plt.show()
#     st.pyplot(fig)
# with col1:
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     cmap = plt.get_cmap('jet')
#     fig=plt.figure(figsize=(7,10),dpi=200)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.coastlines()
#     ax.add_feature(cfeature.STATES)
#     mn=np.around(cc.min(),decimals=-1)
#     mx=np.around(cc.max(),decimals=-1)
#     rng=int((mx-mn)/10)
#     ax.contourf(lonsraa,latsraa,cc[int(cluster_number)-1],levels=np.linspace(mn,mx,rng),cmap=cmap)
#     plt.show()
#     st.pyplot(fig)

# #archive the images for all clusters and have a progress bar that shows image creation
# percent_complete=0
# my_bar = st.progress(0)
# if os.path.exists('/home/ats/midwestprecip/results/'+str(p['n_clusters'])+'_'+level+'_'+classifier_name)==False:
#     os.mkdir('/home/ats/midwestprecip/results/'+str(p['n_clusters'])+'_'+level+'_'+classifier_name)
#     savedir='/home/ats/midwestprecip/results/'+str(p['n_clusters'])+'_'+level+'_'+classifier_name+'/'
#     for x in range(p['n_clusters']):
#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         cmap = plt.get_cmap('jet')
#         fig=plt.figure(figsize=(7,10),dpi=200)
#         ax = plt.axes(projection=ccrs.PlateCarree())
#         ax.coastlines()
#         ax.add_feature(cfeature.STATES)
#         mn=np.around(cc.min(),decimals=-1)
#         mx=np.around(cc.max(),decimals=-1)
#         rng=int((mx-mn)/10)
#         ax.contourf(lonsraa,latsraa,cc[x],levels=np.linspace(mn,mx,rng),cmap=cmap)
#         plt.savefig(savedir+str(x)+'_'+'hgts.png')
        
#         print('imagesave', percent_complete)
#         my_bar.progress(percent_complete + ((x+1)/(p['n_clusters'])))
# else:
#     my_bar.progress(100)
    
                    




    
    