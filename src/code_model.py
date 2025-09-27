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


