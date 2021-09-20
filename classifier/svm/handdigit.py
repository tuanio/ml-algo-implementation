from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

random_state = 12

dataset = load_digits()
X = dataset.data
y = dataset.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=random_state)

svm = SVC(
    kernel='poly',
    degree=4,
    verbose=1,
    random_state=random_state
)
svm.fit(Xtrain, ytrain)

predicted = svm.predict(Xtest)

acc_train = svm.score(Xtrain, ytrain)
acc_test = svm.score(Xtest, ytest)

print(acc_train, acc_test)

cfm = confusion_matrix(ytest, predicted)
sns.heatmap(cfm, annot=True, fmt='.2f')
plt.savefig('hand_digit_cfm.png')