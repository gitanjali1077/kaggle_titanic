import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data= pd.read_csv('train.csv',header=0)
#data.dropna()
data= data.drop(['PassengerId'],axis=1)
print(data.shape)
print(list(data.columns))
print( data.groupby('Survived').mean())

pd.crosstab(data.Pclass,data.Survived).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('pclass')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
plt.show()

pd.crosstab(data.Sex,data.Survived).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
plt.show()

pd.crosstab(data.SibSp,data.Survived).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
plt.show()

pd.crosstab(data.HLTINNOS,data.Survived).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
plt.show()

#print( data.groupby('Fare').mean())


# for findingg the features responsible for seection

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

y=['Survived']
X=[ 'Age','Fare','Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_female',
 'Sex_male', 'Pclass_1', 'Pclass_2' ,'Pclass_3' ,'SibSp_0' ,'SibSp_1', 'SibSp_2',
 'SibSp_3' ,'SibSp_4', 'SibSp_5', 'SibSp_8' ,'Parch_0', 'Parch_1', 'Parch_2',
 'Parch_3', 'Parch_4', 'Parch_5' ,'Parch_6']
print data_final[X].columns.values

logreg = LogisticRegression()
rfe = RFE(logreg, 18)
#try:
rfe = rfe.fit(data_final[X], data_final[y] )
  
print(rfe.support_)
print(rfe.ranking_)



