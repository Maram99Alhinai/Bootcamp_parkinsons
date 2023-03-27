#Read the data
import pandas as pd
df=pd.read_csv('parkinsons.data')


#Get the X and y 
y = df.loc[:,'status']
X = df.drop(['status', 'name'], axis=1)


#Train and fit the model 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)

#test the model
import numpy as np
testSample=np.array([1,2,3,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,9,9,7,7])
dfTest = pd.DataFrame(testSample.reshape(1,-1), columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                                                         'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',      
                                                         'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',      
                                                         'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',       'spread1', 
                                                         'spread2', 'D2', 'PPE'])

testSample_pred = clf.predict(dfTest)
print(testSample_pred)