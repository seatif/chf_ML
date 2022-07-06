import pandas
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
from matplotlib import pyplot

filename = 'evened_data.csv' # change if necessary
df = pandas.read_csv(filename)
df.drop(['Systolic 1', 'Diastolic 1', 'Weight 1', 'Rate 1', 'Oxygen 1', 'Systolic 2', 'Diastolic 2', 'Weight 2', 'Rate 2', 'Oxygen 2', 'Systolic 3', 'Diastolic 3', 'Weight 3', 'Rate 3', 'Oxygen 3'], axis='columns', inplace=True)
input = df.drop('Risk', axis='columns')
target = df['Risk']
input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(input, target, test_size=0.3)

# random forest and corresponding confusion matrix
model = sklearn.ensemble.RandomForestClassifier()
score = -1
for i in range(100):
    model.fit(input_train, output_train)
    s = model.score(input_test, output_test)
    if s > score:
        score = s
print('Random Forest:', score)

y_pred = model.predict(input_test)
cm = sklearn.metrics.confusion_matrix(output_test, y_pred)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 3])
cm_display.plot()
pyplot.show()


# decision tree and corresponding confusion matrix
model = sklearn.tree.DecisionTreeClassifier()
score = -1
for i in range(100):
    model.fit(input_train, output_train)
    s = model.score(input_test, output_test)
    if s > score:
        score = s
print('Decision tree:', score)

y_pred = model.predict(input_test)
cm = sklearn.metrics.confusion_matrix(output_test, y_pred)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 3])
cm_display.plot()
pyplot.show()