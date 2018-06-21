# The needed library's to complete the assignment
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
x_set = iris.data
y_set = iris.target

# This is where we split the data for cross validation
x_set_train, x_set_test, y_set_train, y_set_test = train_test_split(x_set, y_set, test_size=0.2)

model = GaussianNB()
model.fit(x_set_train, y_set_train)
print(model.score(x_set_test, y_set_test))
