import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data= pd.read_csv('train.csv',header=0)
#data.dropna()
data= data.drop(['PassengerId'],axis=1)
data['Age']=data['Age'].fillna(1.0)
data['Age'] = data['Age'].astype(float)
data['Fare'] = data['Fare'].astype(float)


print(data.shape)
print(list(data.columns))
print( data.groupby('Survived').mean())

cat_vars=['Sex','Pclass','SibSp','Parch','Embarked']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['Sex','Pclass','SibSp','Parch','Embarked']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
print data_final.columns.values

data_final_vars=data_final.columns.values.tolist()
cols=['Embarked_S', 'Sex_female',
 'Sex_male', 'Pclass_1','Pclass_3' ,'SibSp_0' ,'SibSp_1', 'SibSp_2',
 'SibSp_3' ,'SibSp_4', 'SibSp_5', 'SibSp_8' ,'Parch_0', 'Parch_1', 'Parch_2',
  'Parch_4', 'Parch_5' ,'Parch_6','Parch_9']

#data_final['Parch_9']=0
X=data_final[cols]
y=data_final['Survived']

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


# This is the testing  data for the prediction
 

datat= pd.read_csv('test.csv',header=0)
#data.dropna()
datat= datat.drop(['Age','Fare'],axis=1)


print(datat.shape)
print(list(datat.columns))
cat_varst=['Sex','Pclass','SibSp','Parch','Embarked']
for var in cat_varst:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(datat[var], prefix=var)
    data1t=datat.join(cat_list)
    datat=data1t
cat_varst=['Sex','Pclass','SibSp','Parch','Embarked']

#print( datat.groupby('Survived').mean())

data_varst=datat.columns.values.tolist()
to_keept=[i for i in data_varst if i not in cat_varst]
data_finalt=datat[to_keept]
print data_finalt.columns.values

data_final_varst=data_finalt.columns.values.tolist()
colst=['Embarked_S', 'Sex_female',
 'Sex_male', 'Pclass_1','Pclass_3' ,'SibSp_0' ,'SibSp_1', 'SibSp_2',
 'SibSp_3' ,'SibSp_4', 'SibSp_5', 'SibSp_8' ,'Parch_0', 'Parch_1', 'Parch_2',
  'Parch_4', 'Parch_5' ,'Parch_6','Parch_9'] 
X1=data_finalt[colst]
print X1
print "test wala data"
#y1=data_finalt['Survived']

# testing data endshere.


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
print logreg.fit(X, y)

y_pred = logreg.predict(X1)
print y_pred
a=np.ceil(y_pred)

np.savetxt("final.csv", a, delimiter=" ")
d= pd.read_csv('final.csv',header=0)
print d
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#kfold = model_selection.KFold(n_splits=10, random_state=7)
#modelCV = LogisticRegression()
#scoring = 'accuracy'
#results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
