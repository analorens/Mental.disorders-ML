#main librarys 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler # Evita que variáveis com escalas diferentes dominem o aprendizado

from sklearn.neural_network import MLPClassifier 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('mental_disorders_formated.csv')

print(df.head()) #show fist lines for understand how the datas is organized
print(df.info()) # general informations about base 
print(df.describe()) #numerical statistics 
print(df.isnull().sum()) 

#Handle missing values
num_cols = df.select_dtypes(include = 'number').columns
for col in num_cols: 
    df[col].fillna(df[col].median(), inplace = True)

    #categorical columns

cat_cols = df.select_dtypes(include = 'object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace = True)

#remove duplicates 

df = df.drop_duplicates()

df = pd.get_dummies (df, columns = cat_cols, drop_first = True)

#Normalization of numerical data

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


