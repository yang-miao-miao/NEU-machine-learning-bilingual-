import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df=pd.read_csv('wpbc.csv')
df.loc[df['Lymph node status'] == '?','Lymph node status'] = 0
df.loc[df['Outcome'] == 'N','Outcome'] = 1#不复发说明好了
df.loc[df['Outcome'] == 'R','Outcome'] = 0#复发说明不好
new_df=pd.concat((df['Tumor size'],df['Lymph node status']),axis=1)
model=KMeans(n_clusters=3).fit(new_df)
new_df=pd.concat((new_df,df['Time'],df['Outcome']),axis=1)
new_df['label']=model.labels_
print(pd.Series(model.labels_).value_counts())
result1=new_df.loc[new_df['label']==0].sort_values('Time')
result2=new_df.loc[new_df['label']==1].sort_values('Time')
result3=new_df.loc[new_df['label']==2].sort_values('Time')
def survive(result):
    a=list(result['Outcome'])
    b=[]
    for i in range(len(a)):
        b.append(str(a[0:(i+1)]).count('1')/(len(b)+1))
    x,y=sorted(np.array(result['Time']))[::-1],sorted(b)
    plt.plot(x,y)
for i in [result1,result2,result3]:
    survive(i)
plt.title('the result of kmeans')
plt.xlabel('time')
plt.ylabel('percent-survival')
plt.show()