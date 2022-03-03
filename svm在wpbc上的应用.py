import  matplotlib.pyplot as plt#导入matplotlib绘图库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,roc_curve,confusion_matrix,classification_report
df=pd.read_csv('wpbc.csv')
df.loc[df['Lymph node status'] == '?','Lymph node status'] = 0
df.loc[df['Outcome']=='N','Outcome']=1
df.loc[df['Outcome']=='R','Outcome']=0
#null_columns=df.columns[df.isnull().any()]
#print(df[df.isnull().any(axis=1)][null_columns])
df['label']=df['Outcome'].values
df.drop(['ID number','Outcome'],inplace=True,axis=1)
print(df)
x,y=df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(df.label.value_counts())
y_train,y_test = y_train.astype('int'),y_test.astype('int')
#sm=SMOTE()
#x_train,y_train=sm.fit_resample(x_train,y_train)
model=svm.SVC(kernel='linear',probability=True)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Accuracy:',accuracy_score(y_test,y_pred))
print('Precision',precision_score(y_test,y_pred))
print('recall',recall_score(y_test,y_pred))
print('f1',f1_score(y_test,y_pred))
print('roc_curve',roc_auc_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
y_score=model.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr)
plt.title('roc_auc')
plt.show()