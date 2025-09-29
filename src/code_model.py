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

#Preparing data for the model

X = df.drop('Mental_Disorder', axis =1)
y = df['Mental_Disorder']

#training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, stratify = y)

#Creating Rede Neural

mlp = MLPClassifier (
    hidden_layer_sizes = (64,32),
    activation = 'relu',
    solver = 'adam',
    max_iter = 500,
    #random_state = 42
)

#training 

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

#Assessment

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Acurácia:", accuracy_score(y_test,y_pred))
print("\nRelatório de Classificação: \n", classification_report(y_test,y_pred))
print("\n Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))


#View Training Neural Network 

import matplotlib.pyplot as plt

plt.figure (figsize=(8,5))
plt.plot(mlp.loss_curve_, marker = 'o', color = 'blue')
plt.litle("Curva de Perda Durante o Treinamento", fontsize = 14)
plt.xlabel("Épocas (iterações)", fontsize = 12)
plt.ylabel("Erro (Loss)", fontsize = 12)
plt.grid(True, linestyle = "--", alpha = 0.7)
plt.show()

#Overfitting check 

train_acc = mlp.score(X_train,y_train)
test_acc = mlp.score(X_test, y_test)

print("Acuracia no treino: {train_acc:.2f}")
print("Acurácia no teste: {test_acc: .2f}")