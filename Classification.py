import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from DataPreparation import Prepare
from sklearn.metrics import classification_report
import pickle

class Classify(Prepare):
  def __init__(self):
      args=['job','marital','education','housing','loan','y']
      self.p = Prepare()
      self.p.read_Data("E:\\NTI\\bank.csv")
      self.x_train,self.x_test,self.y_train,self.y_test = self.p.preProcessing(0.25,args)
      self.lr = LogisticRegression()
      self.knn = KNeighborsClassifier(n_neighbors=5)
      self.naive = GaussianNB()
      self.svc = SVC(kernel="rbf")
      self.dt = DecisionTreeClassifier(criterion='entropy')
      self.rf = RandomForestClassifier(n_estimators=10,criterion='entropy')

  def log_pred(self,features,filename):
      loaded_model = pickle.load(open(filename, 'rb'))
      loaded_model.fit(self.x_train,self.y_train)
      return self.p.make_Predicition(features,loaded_model)

  def model_report(self,model_name):
      report = ""
      acc = ""
      filename = ""
      if model_name == "Logistic Regression":
          self.lr.fit(self.x_train,self.y_train)
          y_pred = self.lr.predict(self.x_test)
          acc = "{0:.2f}%".format(self.lr.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.lr, open(filename, 'wb'))
      elif model_name == "KNN Classifier":
          self.knn.fit(self.x_train,self.y_train)
          y_pred = self.knn.predict(self.x_test)
          acc = "{0:.2f}%".format(self.knn.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.knn, open(filename, 'wb'))
      elif model_name == "Naive Bayes":
          self.naive.fit(self.x_train,self.y_train)
          y_pred = self.naive.predict(self.x_test)
          acc = "{0:.2f}%".format(self.naive.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.naive, open(filename, 'wb'))
      elif model_name == "Support Vector Classifier":
          self.svc.fit(self.x_train,self.y_train)
          y_pred = self.svc.predict(self.x_test)
          acc = "{0:.2f}%".format(self.svc.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.svc, open(filename, 'wb'))
      elif model_name == "Decsion Tree Classifier":
          self.dt.fit(self.x_train,self.y_train)
          y_pred = self.dt.predict(self.x_test)
          acc = "{0:.2f}%".format(self.dt.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.dt, open(filename, 'wb'))
      elif model_name == "Random Forest Classifier":
          self.rf.fit(self.x_train,self.y_train)
          y_pred = self.rf.predict(self.x_test)
          acc = "{0:.2f}%".format(self.rf.score(self.x_test,self.y_test)*100)
          report = classification_report(self.y_test,y_pred)
          filename = model_name+'.sav'
          pickle.dump(self.rf, open(filename, 'wb'))
      else:
          report ="Error in Model Name"
      return acc,report
