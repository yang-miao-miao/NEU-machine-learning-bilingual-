import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
df=pd.read_csv('wpbc.csv')
df.loc[df['Lymph node status'] == '?','Lymph node status'] = 0
df.loc[df['Outcome']=='N','Outcome']=1
df.loc[df['Outcome']=='R','Outcome']=0
#null_columns=df.columns[df.isnull().any()]
#print(df[df.isnull().any(axis=1)][null_columns])
df['label']=df['Outcome'].values
df.drop(['ID number','Outcome'],inplace=True,axis=1)
print(df)
print(df.label.value_counts())
x = df.iloc[:,0:-1].values
y = df.label.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
scar=StandardScaler()
x_train,x_test=scar.fit_transform(x_train),scar.fit_transform(x_test)
import torch
import torch.nn as nn
x_train,y_train=torch.from_numpy(x_train.astype(np.float32)),torch.from_numpy(y_train.astype(np.int64))
x_test,y_test=torch.from_numpy(x_test.astype(np.float32)),torch.from_numpy(y_test.astype(np.int64))
import torch.utils.data as Data
train_data=Data.TensorDataset(x_train,y_train)
train_loader=Data.DataLoader(dataset=train_data,batch_size=30)
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(33,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.hidden(x)
        x=self.classifier(x)
        return x
model=model()
print(model)
from torchviz import make_dot
x=torch.randn(1,33).requires_grad_(True)
y=model(x)
photo=make_dot(y,params=dict(list(model.named_parameters())+[('x',x)]))
photo.view()
optimize=torch.optim.Adam(model.parameters(),lr=0.01)
loss_func=nn.CrossEntropyLoss()
import hiddenlayer as hl
history=hl.History()
canvas=hl.Canvas()
num=20
for epoc in range(60):
    for step,(b_x,b_y) in enumerate(train_loader):
        output=model(b_x)
        train_loss=loss_func(output,b_y)
        optimize.zero_grad()
        train_loss.backward()
        optimize.step()
        time=epoc*len(train_loader)+step+1
        if time%num==0:
            output=model(x_test)
            _,prelab=torch.max(output,1)
            test_accuracy=accuracy_score(y_test,prelab)
            history.log(time,test_accuracy=test_accuracy,train_loss=train_loss)
            with canvas:
                canvas.draw_plot(history['test_accuracy'])
                canvas.draw_plot(history['train_loss'])
plt.show()
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,f1_score,roc_auc_score
import seaborn as sns
confusion=confusion_matrix(y_test,prelab)
print('accuracy',accuracy_score(y_test,prelab))
print('precision',precision_score(y_test,prelab))
print('recall',recall_score(y_test,prelab))
print('f1',f1_score(y_test,prelab))
print('roc_auc',roc_auc_score(y_test,prelab))
print(confusion)
print(classification_report(y_test,prelab))
matrix=pd.DataFrame(confusion)
heatman=sns.heatmap(matrix,annot=True)
plt.show()