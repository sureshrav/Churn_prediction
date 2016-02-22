import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl


# read the data in
df = pd.read_csv("F:\Dataset\churn.csv")

# take a look at the dataset
#print df.columns

colnames=df.columns.tolist()

churnresult=df['Churn']
y=np.where(churnresult=='Yes',1,0)

todrop=['gender']
churnspace=df.drop(todrop,axis=1)

#yesnocols=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"]

#churnspace[yesnocols]=np.where(churnspace[yesnocols] == 'Yes',1)

churnspace=churnspace.replace(['Yes','No','No internet service','No phone service'],[1,0,0,0])
    
churnspace=churnspace.replace(['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'],[1,2,3,4])

churnspace=churnspace.replace(['DSL','Fiber optic'],[1,2])

churnspace=churnspace.replace(['Month-to-month','One year','Two year'],[1,2,3])



print churnspace.head(5)

print "Unique target labels :",np.unique(y)

print np.unique(churnspace['PaymentMethod'])

print np.unique(churnspace['InternetService'])

print np.unique(churnspace['Contract'])

print churnspace.describe()

print churnspace.std()

churnspace.hist()

pl.show()

dummy_pm=pd.get_dummies(churnspace['PaymentMethod'],prefix='PaymentMethod')

print dummy_pm.head()

dummy_is=pd.get_dummies(churnspace['InternetService'],prefix='InternetService')

print dummy_is.head()

dummy_c=pd.get_dummies(churnspace['Contract'],prefix='Contract')

print dummy_c.head()


coltokeep=["customerID","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","MonthlyCharges","TotalCharges","Churn"]

data=churnspace[coltokeep].join(dummy_pm.ix[:,'PaymentMethod_1':])

data=data.join(dummy_c.ix[:,'Contract_1':])

data=data.join(dummy_is.ix[:,'InternetService_0':])

print data.head()

data['intercept']=1.0

train_cols=['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines']

print train_cols

logit=sm.Logit(data['Churn'],data[train_cols])

result=logit.fit()

print result.summary()


