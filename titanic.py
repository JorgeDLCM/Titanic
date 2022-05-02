#Hola, bienvenido al clásico ejercicio de cálculo de la supervivencia del titanic, este es mi ejemplo con 3 modelos de ML
#Puedes inspeccionar mi código, también voy a anexar los data sets csv utilizados
#Saludos

#Importamos librerías Pandas y Numpy para lectura y manipulación del data set

import numpy as np
import pandas as pd

#Importamos los modelos de ML que vamos a usar así como la función train test split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Cargamos el data set y lo convertimos en data frame con pandas

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#Está es la información general del data frame test y train (código comentado para no desplegarse por pantalla)

#print(test.head())
#print(train.head())
#print(test.info())
#print(traom.info())

#Revisión de datos nulos en nuestro data frame (código comentado para no desplegarse por pantalla)

#print(pd.isnull(test).sum())
#print(pd.isnull(train).sum())
#print(test.describe())
#print(train.describe())

#Reemplazo de valores string en columna sexo por 0 para mujeres y 1 para hombres
#Así como un valor numérico para el lugar de embarque (0,1,2)

train['Sex'].replace(['female','male'],[0,1],inplace=True)
test['Sex'].replace(['female','male'],[0,1],inplace=True)

train['Embarked'].replace(['C','S','Q'],[0,1,2],inplace=True)
test['Embarked'].replace(['C','S','Q'],[0,1,2],inplace=True)

#Visualización de la media de edad (código comentado para no desplegarse por pantalla)

#print(train['Age'].mean())
#print(test['Age'].mean())

#Obtención de una media para data frames test y train

promedio = ((train['Age'].mean())+(test['Age'].mean()))/2

#Reemplazo de valores de edad sin datos por la media obtenida

train['Age'] = train['Age'].replace(np.nan, promedio)
test['Age'] = test['Age'].replace(np.nan, promedio)

#Creación de listas bins y names

bins = [0,8,15,18,25,40,60,100]
names = [1,2,3,4,5,6,7]

#Uso de método cut para agrupación de edades en 7 categorías

train['Age'] = pd.cut(train['Age'], bins, labels=names)
test['Age'] = pd.cut(test['Age'], bins, labels=names)

#Verificación de valores vaciós en el data frame (código comentado para no desplegarse por pantalla)

#print(test.info())
#print(pd.isnull(test).sum())
#print(test.info())
#print(pd.isnull(test).sum())

#Eliminación de columna Cabin por su alto contenido de valores vacíos)

train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)

#Eliminación de columnas en data frames que no aportarán al análisis (dejamos la columna PassengerId en el data frame test)

train = train.drop(['PassengerId','Name', 'Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)

#Eliminación de filas con valores vacíos (dato que son pocos los valores vacíos)

train.dropna(axis=0, how='any', inplace=True)
test.dropna(axis=0, how='any', inplace=True)

#Verificación de valores vaciós en el data frame (código comentado para no desplegarse por pantalla)

#print(test.info())
#print(pd.isnull(test).sum())
#print(train.info())
#print(pd.isnull(train).sum())

#Establecimiento de variables x serán nuestros parámetros excluyendo a la columna Survived mientras que y será nuestro parámetro Survived

x = np.array(train.drop(['Survived'], axis=1))
y = np.array(train['Survived'])

#Uso de la función train test split para el ingreso de datos a nuestros modelos de train y test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#Evaluación de un modelo de regresión Logística

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
print('Presición Regresión Logística: ')
print(logreg.score(x_train, y_train))

#Evaluación de un modelo de Clasificación de Vector de Soporte

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('Presición Soporte de Vectores: ')
print(svc.score(x_train, y_train))

#Evaluación de un modelo del vecino más cercano

knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(x_train, y_train)
y_pred = knc.predict(x_test)
print('Presición Vecino más Cercano: ')
print(knc.score(x_train, y_train))

#Estanlecemos la columna PassengerId del dataframe para hacer la predicción de nuestros modelos establecidos y evaluados
#Se crean nuevos data frames con la predicción de cada modelo dónde podemos comparar la predicción con los datos de entrenamiento

ids = test['PassengerId']

prediccion_logreg = logreg.predict(test.drop('PassengerId',axis=1))
out_logreg = pd.DataFrame({'PassengerId':ids, 'Survived':prediccion_logreg})
print('Presición Regresión Logística: ')
print(out_logreg.head())

prediccion_svc = svc.predict((test.drop('PassengerId',axis=1)))
out_svc = pd.DataFrame({'PassengerId':ids, 'Survived':prediccion_svc})
print('Presición Soporte de Vectores: ')
print(out_svc.head())

prediccion_knc = knc.predict((test.drop('PassengerId',axis=1)))
out_knc = pd.DataFrame({'PassengerId':ids, 'Survived':prediccion_knc})
print('Presición Vecino más Cercano: ')
print(out_knc.head())