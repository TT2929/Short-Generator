import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import joblib


def study_model():

    df = pd.read_csv(output_folder / "logs" / "labeled_log.csv")

    X = df[[col for col in df.columns if col.startswith("sem_")]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("classification report", classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, y_proba))

    joblib.dump(clf, "funny_pred_model.pkl")
    return print("COMPLETE:Study model")


#base_directory
base_directory = Path("")

#output-folder
output_folder = Path( base_directory / "Output" )
output_folder.mkdir(exist_ok=True)

df = pd.read_csv(output_folder / "logs" / "labeled_log.csv")

X = df[[col for col in df.columns if col.startswith("sem_")]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=1)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("classification report", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))

joblib.dump(clf, "funny_pred_model.pkl")
