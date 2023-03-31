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
    testSample=np.array([float(request.form["MDVP:Fo"]),float(request.form["MDVP:Fhi"]),float(request.form["MDVP:Flo"])
                         ,float(request.form["MDVP:Jitter"]),float(request.form["MDVP:Jitter(Abs)"]),float(request.form["MDVP:RAP"])
                         ,float(request.form["MDVP:PPQ"]),float(request.form["Jitter:DDP"]),float(request.form["MDVP:Shimmer"])
                         ,float(request.form["MDVP:Shimmer(dB)"]),float(request.form["Shimmer:APQ3"]),float(request.form["Shimmer:APQ5"])
                         ,float(request.form["MDVP:APQ"]),float(request.form["Shimmer:DDA"]),float(request.form["NHR"]),
                         float(request.form["HNR"]),float(request.form["RPDE"]),float(request.form["DFA"])
                         ,float(request.form["spread1"]),float(request.form["spread2"]),float(request.form["D2"]),
                         float(request.form["PPE"])])
    
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
 