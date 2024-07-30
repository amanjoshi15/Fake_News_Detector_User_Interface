import numpy as np
import pickle
from flask import Flask, request, render_template

app=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))
cv=pickle.load(open("count_vectorizer.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    title=request.form['title']
    description=request.form['description']
    data=[title,description]
    features = cv.transform(data).toarray()
    prediction=model.predict(features)
    output = 'not fake' if prediction[0] == 1 else 'fake'
    return render_template("index.html",prediction_text="This news is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)