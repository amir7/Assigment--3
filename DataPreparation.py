import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split



class Prepare:

  def __init__(self):
      self.job = LabelEncoder()
      self.marital = LabelEncoder()
      self.education = LabelEncoder()
      self.housing = LabelEncoder()
      self.loan = LabelEncoder()
      self.dependant = LabelEncoder()
      self.sc = StandardScaler()

  def read_Data(self,path):
      self.df = pd.read_csv(path,delimiter=";")
      self.df = self.df.drop(["pdays","previous","poutcome","default","month","day","contact"],axis=1)


  def label__(self):
      self.df["job"] = self.job.fit_transform(self.df["job"].values)
      self.df['marital'] = self.marital.fit_transform(self.df['marital'].values)
      self.df['education'] = self.education.fit_transform(self.df['education'].values)
      self.df['housing'] = self.housing.fit_transform(self.df['housing'].values)
      self.df['loan'] = self.loan.fit_transform(self.df['loan'].values)
      self.df['y'] = self.dependant.fit_transform(self.df['y'].values)

  def split(self):
      self.x = self.df.iloc[:,:-1].values
      self.y = self.df.iloc[:,-1].values

  def normalize(self):
      self.x = self.sc.fit_transform(self.x)

  def train_test(self,test_size):
      self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size = test_size)
      return self.x_train,self.x_test,self.y_train,self.y_test

  def preProcessing(self,testSize,labels):
      self.label__()
      self.split()
      self.normalize()
      self.x_train,self.x_test,self.y_train,self.y_test =self.train_test(testSize)
      return self.x_train,self.x_test,self.y_train,self.y_test

  def make_Predicition(self,features,clf):
      features[0] = int(features[0])
      features[4] = int(features[4])
      features[7] = int(features[7])
      features[8] = int(features[8])
      features[1] = self.job.transform([features[1]])
      features[2] = self.marital.transform([features[2]])
      features[3] = self.education.transform([features[3]])
      features[5] = self.housing.transform([features[5]])
      features[6] = self.loan.transform([features[6]])
      scaling = self.sc.transform(np.array(features).reshape(1,-1))
      return str(self.dependant.inverse_transform(clf.predict(scaling)[0]))
