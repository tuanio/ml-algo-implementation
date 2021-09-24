from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from pydotplus import graph_from_dot_data
import pandas as pd
import matplotlib.pyplot as plt

random_state = 12

data = fetch_openml("titanic", version=1, as_frame=True)
df = pd.DataFrame(data.data)
df['target'] = data.target

# drop all na
df.dropna(axis=1, inplace=True)

encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])

df.drop(['name', 'ticket'], axis=1, inplace=True)

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=random_state)

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(Xtrain, ytrain)

acc_train = dt.score(Xtrain, ytrain)
acc_test = dt.score(Xtest, ytest)

print(acc_train, acc_test)


dot_data = export_graphviz(dt, filled=True, rounded=True, class_names=df['target'].values.unique(), feature_names=list(df.columns[:-1]), out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('titanic_tree.png')

# plot_tree(dt)
# plt.savefig('titanic_tree.png')