from utils import db_connect
engine = db_connect()



import pandas as pd

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")
total_data.head()

X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators = 200, learning_rate = 0.001, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

