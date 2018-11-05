from flask import Flask
from Classification import Classify
from flask import request,render_template, url_for
import os
import pickle

m = Classify()
app = Flask(__name__)

IMAGES_FOLDER = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

@app.route('/')
def root():
    intro = "<p> The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.</p>"
    link = "<p> https://archive.ics.uci.edu/ml/datasets/Bank+Marketing</p>"
    return intro+link

@app.route('/visual')
def show_image():
    im1 = os.path.join(app.config['UPLOAD_FOLDER'], 'im1.png')
    im2 = os.path.join(app.config['UPLOAD_FOLDER'], 'im2.png')
    im3 = os.path.join(app.config['UPLOAD_FOLDER'], 'im3.png')
    im4 = os.path.join(app.config['UPLOAD_FOLDER'], 'im4.png')
    im5 = os.path.join(app.config['UPLOAD_FOLDER'], 'im5.png')
    im6 = os.path.join(app.config['UPLOAD_FOLDER'], 'im6.png')
    return render_template("visual.html", image1 = im1, image2=im2, image3 = im3,image4 = im4,image5=im5,image6=im6)

@app.route('/classification',methods=['GET'] )
def classification_fun():
    model_name = request.args.get("browser")
    acu = ""
    acc,report = m.model_report(model_name)
    if model_name == "Logistic Regression":
        acu = " Logistec Regression Accuracy is "
    elif model_name == "KNN Classifier":
        acu = " KNN Classifier Accuracy is "
    elif model_name == "Naive Bayes":
        acu = " Naive Bayes Accuracy is "
    elif model_name == "Support Vector Classifier":
        acu = " Support Vector Classifier  Accuracy is "
    elif model_name == "Decsion Tree Classifier":
        acu = " Decsion Tree Classifier Accuracy is "
    elif model_name == "Random Forest Classifier":
        acu = " Random Forest Classifier Accuracy is "
    else:
        acu ="Error in Model Name"
    acu +=acc+"<p>Model Report</p>"+"<p>"+report+"</p>"
    return acu


@app.route('/predict')
def student():
    return render_template('prediction.html')

@app.route('/train')
def train():
    return render_template('list.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      Age = request.form['Age']
      Job = request.form['Job']
      Marital = request.form['Marital']
      Education = request.form['Education']
      Balance = request.form['Balance']
      Housing = request.form['Housing']
      Loan = request.form['Loan']
      Duration = request.form['Duration']
      Campaign = request.form['Campaign']
      features = [Age,Job,Marital,Education,Balance,Housing,Loan,Duration,Campaign]
      return m.log_pred(features,"logistic.sav")

if __name__ == "__main__":
    try:
        app.run(debug=True,port='5000')
    except Exception as e:
        print("Error")
