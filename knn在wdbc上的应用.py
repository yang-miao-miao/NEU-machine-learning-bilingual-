import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,roc_curve,confusion_matrix,classification_report
df=pd.read_csv('wdbc.csv')
df.loc[df['Diagnosis']=='M','Diagnosis']=1
df.loc[df['Diagnosis']=='B','Diagnosis']=0
df['label']=df['Diagnosis'].values
df.drop(['ID number','Diagnosis'],inplace=True,axis=1)
print(df)
x,y=df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
y_train,y_test = y_train.astype('int'),y_test.astype('int')
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuracy:',accuracy_score(y_test,y_pred))
print('Precision',precision_score(y_test,y_pred))