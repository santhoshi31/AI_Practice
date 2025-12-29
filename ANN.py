from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

iris = load_iris()
x = iris.data
y = (iris.target == 0).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = Sequential()
# first hidden layer 6 nutrons 4 input
model.add(Dense(units=6, activation='relu', input_dim = 4)) # 4 feature in the iris dataset,6 nerones
model.add(Dense(units=6, activation='relu'))# another hidden layer is 6 nutrons
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'] )
model.summary()
model.fit(x_train, y_train, batch_size=8, epochs=25, validation_split=0.2)
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy:{accuracy:.4f}")
