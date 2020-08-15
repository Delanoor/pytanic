def predict(pclass,sex,age,sibsp,parch,fare,embarked,title):

    import pickle
    x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    random_forest = pickle.load(open('titanic/titanic_model.sav','rb'))
    prediction = random_forest.predict(x)
    if prediction == 0:
        prediction = 'Uh oh'
    elif prediction == 1:
        prediction = "Survived !!"
    else:
        prediction = "Error"
    return prediction
