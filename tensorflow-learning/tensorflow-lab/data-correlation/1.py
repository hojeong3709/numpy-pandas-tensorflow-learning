import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = np.random.randint(1, 100, size=[100, 2])
y = np.random.randint(0, 3, size=100)

e = np.eye(3)
y_one_hot = e[y]

df = pd.DataFrame(x, columns=["x1", "x2"])
df["target"] = y
print(df)
'''
r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
'''
print(df.corr())

# sns.heatmap(df.corr(), vmin=-1, vmax=1,
#             linewidths=0.5, annot=True,
#             cmap=plt.cm.gist_heat)

# sns.pairplot(df)

sns.pairplot(df, vars=["x1", "x2"], hue="target")
plt.show()

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(units=3, input_dim=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.1), metrics=['accuracy'])
model.fit(x, y_one_hot, epochs=200)
