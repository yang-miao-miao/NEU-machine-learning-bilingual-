import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
df=pd.read_csv('wdbc.csv')
df.loc[df['Diagnosis']=='M','Diagnosis']=1
df.loc[df['Diagnosis']=='B','Diagnosis']=0
label=df['Diagnosis'].values
df['label']=label
df.drop(['ID number','Diagnosis'],inplace=True,axis=1)
print(df)
from sklearn.decomposition import PCA
pca_line = PCA().fit(df)
print('特征值:','\n',pca_line.explained_variance_)
print('比例','\n',pca_line.explained_variance_ratio_)
plt.plot(np.arange(1,len(df.columns)+1),np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()
pca=PCA(3)
pca.fit(df)
new_df=pca.transform(df)
new_df=pd.DataFrame(new_df)
new_df['label']=label
print(new_df)
model=KMeans(n_clusters=3).fit(new_df)
new_df['label']=model.labels_
print(pd.Series(model.labels_).value_counts())
class1=new_df[new_df['label']==0]
class2=new_df[new_df['label']==1]
class3=new_df[new_df['label']==2]
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(class1[0],class1[1],class1[2])
ax.scatter(class2[0],class2[1],class2[2])
ax.scatter(class3[0],class3[1],class3[2])
plt.show()
