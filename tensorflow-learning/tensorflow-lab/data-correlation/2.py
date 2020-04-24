import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']
print(df)

print(df.corr())

sns.heatmap(df.corr(), vmin=-1, vmax=1,
            linewidths=0.5, annot=True,
            cmap=plt.cm.gist_heat)
# plt.show()

sns.pairplot(df)
# plt.show()

sns.pairplot(vars=['petal length (cm)',
                   'petal width (cm)', 'target'],
             hue='target', data=df)
plt.show()
