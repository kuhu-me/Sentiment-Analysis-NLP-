''' IMPORTING ALL THE LIBRARIES'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import vectorizer

param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'liblinear'],  # Solvers that work with small datasets
    'max_iter': [10000, 20000, 40000]
}

model = LogisticRegression( class_weight='balanced')

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=1
)

print("\n\nFitting GridSearchCV...\n\n")
grid.fit(vectorizer.X_train, vectorizer.Y_train)

best_model = grid.best_estimator_
predicted = best_model.predict(vectorizer.X_train)
accuracy = accuracy_score(vectorizer.Y_train, predicted)

print("Accuracy:", accuracy)
print("Best parameters:", grid.best_params_)


