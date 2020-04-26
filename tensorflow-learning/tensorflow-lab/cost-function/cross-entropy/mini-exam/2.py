import numpy as np
import tensorflow as tf
from sklearn import model_selection

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

data = np.loadtxt("../../../data/sonar.csv", dtype=np.str, delimiter=",")

y_data = data[:, -1:]
y_data_list = list()
for n in y_data:
    if n == 'R':
        y_data_list.append(0)
    else:
        y_data_list.append(1)

# m = map(lambda n: 0 if n == "R" else 1, data[:, -1])

x_data = np.float32(data[:, :-1])
y_data = np.array(y_data_list, dtype=np.float32).reshape(len(y_data_list), 1)
# print(y_data)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)
# print(x_train)
# print(x_test)

print(y_train.shape)
print(y_test.shape)


IO = Dense(units=1, input_shape=[60], activation="sigmoid")
model = Sequential([IO])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
earlyStop = EarlyStopping(monitor="loss", patience=20, min_delta=0.001)
hisotry = model.fit(x_train, y_train, epochs=10000, callbacks=[earlyStop])

# sigmoid를 적용한 후 cast까지 완료해서 결과 전달
predict = model.predict_classes(x_test)
print(predict)
model.evaluate(x_test, y_test)








