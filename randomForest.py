import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier



crimes = pd.read_csv('selected_features_2012_to_2017.csv',error_bad_lines=False)
#crimes = crimes.loc[crimes['Location Description Number'] != 0]
crimes.index = pd.DatetimeIndex(crimes.Date)
print(crimes.head(),len(crimes))
print (crimes.info())

crimes_2012_2016 = crimes.loc['2012':'2016']
crimes_2017 = crimes.loc['2017']
print('Number of observations in the training data:', len(crimes_2012_2016))
print('Number of observations in the test data:',len(crimes_2017))

#pri_type = crimes_2012_2016[crimes_2012_2016.columns[1:33]]
features_train = crimes_2012_2016[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
#features = crimes_2012_2016[["hour", "Day of Week", "Community Area", "Business Hour", "Business Day"]]
#features = pd.concat([pri_type, features], axis=1)
y_train = crimes_2012_2016["Location Description Number"]

#pri_type_test = crimes_2017[crimes_2017.columns[1:33]]
features_test = crimes_2017[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
#features_test = crimes_2017[["hour", "Day of Week", "Community Area", "Business Hour", "Business Day"]]
#features_test = pd.concat([pri_type_test, features_test], axis=1)
y_test = crimes_2017["Location Description Number"]

time_start = time.clock()

# build random forest model
trees = 35
depth = 15
clf = RandomForestClassifier(n_estimators=trees, max_depth=depth)
clf.fit(features_train, y_train)
peds = clf.predict(features_test)
comparison = pd.crosstab(y_test, peds)

# # build gradient boosting trees model
# trees = 100
# depth = 10
# clf = GradientBoostingClassifier(n_estimators=trees, max_depth=depth)
# clf.fit(features_train, y_train)
# peds = clf.predict(features_test)
# comparison = pd.crosstab(y_test, peds)

# print(peds[0:10])
# print(crimes_2017["Location Description Number"].head(10))
# print(comparison)
# print (clf.feature_importances_)
# print (clf.score(features_test, y_test))
# print (clf.score(features_train, y_train))
# scores = cross_val_score(clf, features_test, y_test)
# print (scores)



time_elapsed = (time.clock() - time_start)
print("time to build a model is:", time_elapsed)

classificationReport = classification_report(y_test, peds, digits=5)
print(classificationReport)

class_names = [1,2,3]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, peds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()