import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
from torchviz import make_dot
import hiddenlayer as hl
df=pd.read_csv('wdbc.csv')
df.loc[df['Diagnosis']=='M','Diagnosis']=1
df.loc[df['Diagnosis']=='B','Diagnosis']=0
#null_columns=df.columns[df.isnull().any()]
#print(df[df.isnull().any(axis=1)][null_columns])
df['label']=df['Diagnosis'].values
df.drop(['ID number','Diagnosis'],inplace=True,axis=1)
print(df)
x,y=df.iloc[:,:-1].values,df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
scar=StandardScaler()
x_train,x_test=scar.fit_transform(x_train),scar.transform(x_test)
x_train,y_train=torch.from_numpy(x_train.astype(np.float32)),torch.from_numpy(y_train.astype(np.int64))
x_test,y_test=torch.from_numpy(x_test.astype(np.float32)),torch.from_numpy(y_test.astype(np.int64))

train_data=Data.TensorDataset(x_train,y_train)
test_data=Data.TensorDataset(x_test,y_test)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(30,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(5,2),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.hidden(x)
        x=self.classifier(x)
        return x
model=model()
print(model)

x=torch.randn(1,30).requires_grad_(True)
y=model(x)
photo=make_dot(y,params=dict(list(model.named_parameters())+[('x',x)]))
#photo.view()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
lossfuc=nn.CrossEntropyLoss()

history=hl.History()
canvas=hl.Canvas()
epocstep=20
for epoc in range(60):
    for step,(b_x,b_y) in enumerate(train_loader):
        output=model(b_x)
        train_loss=lossfuc(output,b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        time=epoc*len(train_loader)+step+1
        if time%epocstep==0:
            output=model(x_test)
            _,prelab=torch.max(output,1)
            test_accuracy=accuracy_score(y_test,prelab)
            history.log(time,train_loss=train_loss,test_accuracy=test_accuracy)
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