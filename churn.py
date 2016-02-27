import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl

df = pd.read_csv("F:\Dataset\churn.csv")

colnames=df.columns.tolist()

churnresult=df['Churn']

y=np.where(churnresult=='Yes',1,0)

churnspace=df

churnspace=churnspace.replace(['Yes','No','No internet service','No phone service'],[1,0,0,0])
    
churnspace=churnspace.replace(['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'],[1,2,3,4])

churnspace=churnspace.replace(['DSL','Fiber optic','Male','Female'],[1,2,1,0])

churnspace=churnspace.replace(['Month-to-month','One year','Two year'],[1,2,3])

#print churnspace.head(5)

#print "Unique target labels :",np.unique(y)

#print np.unique(churnspace['PaymentMethod'])

#print np.unique(churnspace['InternetService'])

#print np.unique(churnspace['Contract'])

#print churnspace.describe()

#print churnspace.std()

#churnspace.hist()

#pl.show()

dummy_pm=pd.get_dummies(churnspace['PaymentMethod'],prefix='PaymentMethod')

#print dummy_pm.head()

dummy_is=pd.get_dummies(churnspace['InternetService'],prefix='InternetService')

#print dummy_is.head()

dummy_c=pd.get_dummies(churnspace['Contract'],prefix='Contract')

#print dummy_c.head()

coltokeep=["customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","MonthlyCharges","TotalCharges","Churn"]

data=churnspace[coltokeep].join(dummy_pm.ix[:,'PaymentMethod_1':])

data=data.join(dummy_c.ix[:,'Contract_1':])

data=data.join(dummy_is.ix[:,'InternetService_0':])

#print data.head()

data['intercept']=1.0

x=data['TotalCharges']

x=x.convert_objects(convert_numeric=True)

#print x

todrop=['TotalCharges','Churn']

t=data.drop(todrop,axis=1)

#print t.dtypes

t=pd.concat([t,x],axis=1)

print t.dtypes

print t.head(9)

todrop=['TotalCharges','MonthlyCharges']

t=t.drop(todrop,axis=1)

print t.dtypes

cols=t.columns

print cols
'''

t[cols]=t[cols].astype(int)'''

print t.dtypes

print data.dtypes

m=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','MonthlyCharges','intercept']

m=data[m]

print m

logit=sm.Logit(data['Churn'],m)

result=logit.fit()

print result.summary()

print result.conf_int()


