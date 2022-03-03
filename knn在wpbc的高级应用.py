import  matplotlib.pyplot as plt#导入matplotlib绘图库
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,roc_curve,confusion_matrix,classification_report,make_scorer
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
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#print(df.label.value_counts())
y_train,y_test = y_train.astype('int'),y_test.astype('int')
model=KNeighborsClassifier()
parameters = {'n_neighbors':[2,3,4,5,6,7,8,9,10],
              'weights':['uniform','distance'],
              'leaf_size':[15,20,25,30,35,40,45]
}
scorers={
    'precision_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'accuracy_score':make_scorer(accuracy_score)
}
#这个参数可以自己调，模型默认使用训练结果最优参数重新训练
refit_score='precision_score'
grid_search=GridSearchCV(model,parameters,refit=refit_score,cv=3,return_train_score=True,scoring=scorers,n_jobs=-1)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print(grid_search.best_params_)
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pre_yes','pre_no'],index=['yes','no']))
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("f1",f1_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("roc_auc",roc_auc_score(y_test,y_pred))
y_score=grid_search.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,label="DecisionTreeClassifier_auc")
plt.title('roc_auc')
plt.legend()
plt.grid()
plt.show()