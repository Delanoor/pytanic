from keras.models import load_model
def predict(pclass,sex,age,sibsp,parch,fare,embarked,title):

    X = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    model_predict = load_model('titanic_NN.h5')
    prediction = model_predict.predict(X)
    if prediction < 0.5:
        prediction = 'Uh oh'
    else:
        prediction = "Survived !!"
    return prediction
