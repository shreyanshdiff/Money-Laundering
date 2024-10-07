import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , r2_score , precision_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/ML.csv')
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['typeofaction'] = encoder.fit_transform(df['typeofaction'])
df['typeoffraud'] = encoder.fit_transform(df['typeoffraud'])

X = df[['typeofaction','sourceid','destinationid','amountofmoney','hour','day_of_week','month']]
y = df['isfraud']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state=42) 
xgb_model = XGBClassifier(random_state = 42 , eval_metric = 'mlogloss')

param_grid = {
    'n_estimators':[100,200],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.7,0.8,1.0],
    'colsample_bytree':[0.7,0.8,1.0]
} 

grid_search = GridSearchCV(estimator = xgb_model , param_grid = param_grid , cv= 3 , scoring = 'accuracy' , verbose = 1 , n_jobs = -1)
grid_search.fit(X_train , y_train)

best_model = grid_search.best_estimator_
print("Best Model Parameters:" , grid_search.best_params_)
print("Best Model Parameters:" , grid_search.best_params_)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

from sklearn.metrics import confusion_matrix , accuracy_score , r2_score , precision_score , classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score , r2_score , precision_score
print("Accuracy" , accuracy_score(y_pred , y_test))
print("R2" , r2_score(y_pred , y_test))
print("Precision" , precision_score(y_pred , y_test))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(best_model, 'xgb.pkl')
print("Model saved to fraud_detection_model_xgb.pkl")
