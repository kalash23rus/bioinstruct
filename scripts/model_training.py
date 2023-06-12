I have refactored the code and split it into three modules: `data.py`, `models.py`, and `main.py`. 

`data.py` contains the code to split the data into training and testing sets.
```python
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
```

`models.py` contains the code to train and evaluate the models.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # ... (same as the original code)

def save_leaderboard_and_models(leaderboard, trained_models, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((leaderboard, trained_models), f, protocol=4)


`main.py` contains the main script to run the code.

from data import split_data
from models import train_and_evaluate_models, save_leaderboard_and_models

X_train, X_test, y_train, y_test = split_data(X, y)

leaderboard, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

save_leaderboard_and_models(leaderboard, trained_models, '/home/a.kalashnikov/projects/chatGPT/qa_datasets_generation/good_bad_qestion_classificator/models/leaderboard_models.pkl')
