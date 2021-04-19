from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def cnn_regressor(X, y):
    X = X.to_numpy().reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    
    print("Model Summary", model.summary(), "\n")

    model.fit(X_train, y_train, batch_size=12,epochs=200, verbose=0)
    print("Evaluate model", model.evaluate(X_train, y_train), "\n")

    predicts = model.predict(X_test)

    mse = mean_squared_error(y_test, predicts)
    print('Mean Absolute Error:', mean_absolute_error(y_test, predicts))
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', np.sqrt(mse))
    print('Mean Absolute Percentage Error:', np.mean(np.abs((y_test,predicts.reshape(predicts.shape[0])))*100))
    print('R2:', r2_score(y_test, predicts))