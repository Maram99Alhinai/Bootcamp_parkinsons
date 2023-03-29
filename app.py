from flask import Flask, render_template, request
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    df=pd.read_csv('parkinsons.data')
    y = df.loc[:,'status']
    X = df.drop(['status', 'name'], axis=1)
    clf = GaussianNB()
    clf.fit(X, y)
    testSample=np.array([int(request.form["MDVP:Fo"]),int(request.form["MDVP:Fhi"]),int(request.form["MDVP:Flo"])
                         ,int(request.form["MDVP:Jitter"]),int(request.form["MDVP:Jitter(Abs)"]),int(request.form["MDVP:RAP"])
                         ,int(request.form["MDVP:PPQ"]),int(request.form["Jitter:DDP"]),int(request.form["MDVP:Shimmer"])
                         ,int(request.form["MDVP:Shimmer(dB)"]),int(request.form["Shimmer:APQ3"]),int(request.form["Shimmer:APQ5"])
                         ,int(request.form["MDVP:APQ"]),int(request.form["Shimmer:DDA"]),int(request.form["NHR"]),
                         int(request.form["HNR"]),int(request.form["RPDE"]),int(request.form["DFA"])
                         ,int(request.form["spread1"]),int(request.form["spread2"]),int(request.form["D2"]),
                         int(request.form["PPE"])])
    
    dfTest = pd.DataFrame(testSample.reshape(1,-1), columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                                                         'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',      
                                                         'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',      
                                                         'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',       'spread1', 
                                                         'spread2', 'D2', 'PPE'])
    testSample_pred = clf.predict(dfTest)
    if testSample_pred==1:
        return  f"parkinsons"
    if testSample_pred==0:
        return f"healthy"
if __name__ == '__main__':
    app.run(debug=True)
 