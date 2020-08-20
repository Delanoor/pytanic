#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
df = pd.read_csv("train.csv")

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# A list with the all the different titles
titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))


# Normalize the titles
def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ["Jonkheer","Don",'the Countess', 'Dona', 'Lady',"Sir"]:
        return 'Royalty'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

# Lets create a new column for the titles
df['Title'] = df['Name'].map(lambda x: get_title(x))
# train.Title.value_counts()
# train.Title.value_counts().plot(kind='bar')

# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'
df['Title'] = df.apply(replace_titles, axis=1)


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna("S", inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df.drop("Name", axis=1, inplace=True)
df.Sex.replace(('male','female'), (0,1), inplace = True)
df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)


X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df["Survived"]

model = Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_train = model.fit(X, y, epochs=500, batch_size=50, verbose=0,
                validation_split=0.06)

model.save('titanic_NN.h5')
