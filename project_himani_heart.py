import pandas as pd
df= pd.read_csv('heart.csv')
print(df.head(2))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled  = scaler.transform(X_test)       # only transform

import pandas as pd

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

from sklearn.linear_model import LogisticRegression

# Model create
log_model = LogisticRegression(max_iter=1000)

# Model train
log_model.fit(X_train, y_train)

# Prediction
y_pred_log = log_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_log))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_log))

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    pipeline.set_params(knn__n_neighbors=k)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,                # 5-Fold Cross Validation
        scoring="accuracy"
    )
    cv_scores.append(scores.mean())
best_k = k_range[np.argmax(cv_scores)]
best_score = max(cv_scores)

print("Best k:", best_k)
print("Best Cross-Validation Accuracy:", best_score)
for k, score in zip(k_range, cv_scores):
    print(f"k={k}, CV Accuracy={score:.4f}")

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=best_k)  
model_knn.fit(X_train, y_train)

y_pred_knn = model_knn.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_knn))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

from sklearn.svm import SVC

# SVM Model (Linear Kernel)
svm_model = SVC(kernel='linear', random_state=42)

# Train model
svm_model.fit(X_train, y_train)

# Prediction
y_pred_svm = svm_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Confusion Matrix
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# Classification Report
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

from sklearn.tree import DecisionTreeClassifier

# Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion='gini',      
    max_depth=5,           
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Prediction
y_pred_dt = dt_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Confusion Matrix
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Classification Report
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

# Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Confusion Matrix
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Classification Report
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
import seaborn as sns

#for target variable to check data is balanced or not with the help of matplotlib
plt.figure()
df['target'].value_counts().plot(kind='bar')
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Patient Count")
plt.title("Target Distribution")
plt.show()

# for age and target to check heart disease chances according to age with the help of matplotlib
plt.figure()

plt.hist(
    [df[df['target'] == 0]['age'], df[df['target'] == 1]['age']],
    label=['No Disease', 'Disease'],
    alpha=0.5,
    edgecolor='black'
)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age vs Heart Disease")
plt.legend()
plt.show()

#for thalach and target to check heart disease according to low thalach (weak heart) with the help of matplotlib
plt.figure()

plt.hist(
    [df[df['target'] == 0]['thalach'], df[df['target'] == 1]['thalach']],
    label=['No Disease', 'Disease'],
    alpha=0.5,
    edgecolor='black'
)

plt.xlabel("thalach")
plt.ylabel("No. of People.")
plt.title("Thalach vs Heart Disease")
plt.legend()
plt.show()

#for cp and target to check heart disease accoridng to chest pain type with the help of seaborn
plt.figure()
sns.countplot(x='cp', hue='target', data=df)
plt.xlabel("Chest Pain Type Level")
plt.ylabel("Patient Count")
plt.title("Chest Pain vs Heart Disease")
plt.show()

