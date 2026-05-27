from sklearn import datasets

wdbc = datasets.load_breast_cancer()

X = wdbc.data
y = wdbc.target

print("Data shape", X.shape)
print("Target shape", y.shape)
print("Target names", wdbc.target_names)
print("First 5 feature names", wdbc.feature_names[:5])

print("="*20)

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

model=svm.SVC()
model.fit(X,y)
predict=model.predict(X)
accuracy = accuracy_score(y, predict)
precision = precision_score(y, predict)
recall = recall_score(y, predict)
print("Accuracy", accuracy)
print("Precision", precision)
print("Recall", recall)

print("="*20)

from sklearn import svm, tree, neighbors, ensemble

models = {
    "SVM" : svm.SVC(),
    "Decision Tree" : tree.DecisionTreeClassifier(random_state=42),
    "KNN" : neighbors.KNeighborsClassifier(),
    "Random Forest" : ensemble.RandomForestClassifier(random_state=42)
}

results = {}

for i in models:
    model = models[i]
    model.fit(X,y)
    predict=model.predict(X)
    accuracy = accuracy_score(y, predict)
    precision = precision_score(y, predict)
    recall = recall_score(y, predict)
    print(i,"Accuracy", accuracy)
    results[i] = accuracy

print("="*20)

print("Best model : ", max(results, key=results.get))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

models[max(results, key=results.get)].fit(X,y)
predict=models[max(results, key=results.get)].predict(X)
cm = confusion_matrix(y, predict)
print("Confusion Matrix")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wdbc.target_names)
disp.plot()
plt.show()
plt.savefig("confusion_matrix.png")


scatter = plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel(wdbc.feature_names[0])
plt.ylabel(wdbc.feature_names[1])
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()
plt.savefig("scatter_plot.png")