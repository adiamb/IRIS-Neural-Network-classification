from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical


data = load_iris()
X_all = data.data
y_all = to_categorical(data.target)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(4, input_dim=4))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(3)) 
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

hist=model.fit(X_tr, y_tr, epochs = 500, validation_data=(X_te, y_te))



# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
