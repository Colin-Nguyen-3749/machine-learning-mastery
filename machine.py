# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Check dimensions of the dataset
# print("Dimensions of dataset: " + str(dataset.shape))

# Peek at the dataset
# print(dataset.head(20))

# Get summary statistics
# print(dataset.describe())

# Look at the amount of rows in each class
# print(dataset.groupby('class').size())

# create box and whisker plots to visualize each attribute
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

#========================================

# Create validation dataset (80% of iris dataset)
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Next, use a stratified 10-fold cross validation to estimate model accuracy
# This splits our dataset into 10, where it will be trained on 9 of the splits 
# and then tested on 1 split
# Afterwards, repeat this pattern, but choose a new split to be tested on each time
# The purpose of our test splits being stratified is to ensure that each split
# of the dataset will have an even distribution of observations
# For example, if we had a dataset of pets, we wouldn't want one split to continue 
# mostly dogs and another mostly cats, we'd like an even distribution of them all

# For the random seed in random_state, let this be a fixed number to make sure that 
# each algorithm is evaluated on the same splits of the training dataset

# Evaluate models by using accuracy
# This is the ratio of the predicted instances that were predicted correctly divided by 
# all other predictions (right and wrong), with the final result being multiplied by 100

# Build all models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate all of them
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))