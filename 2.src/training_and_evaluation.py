import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, make_scorer

def train_decision_tree(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save):
    tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    tree_clf.fit(X_train_scal_re, y_train_balanced)
    predictions_tree_clf = tree_clf.predict(X_test_scal)
    print("Decision Tree Test Accuracy:", round(tree_clf.score(X_test_scal, y_test), 4))
    print(classification_report(y_test, predictions_tree_clf))
    print("Precision:", round(precision_score(y_test, predictions_tree_clf, average='macro'), 4))
    print("Recall:", round(recall_score(y_test, predictions_tree_clf, average='macro'), 4))
    with open(path_save + 'Tree_clf_trained_model.pkl', 'wb') as file:
        pickle.dump(tree_clf, file)
    return tree_clf

def train_random_forest(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save):
    class_weight_op1 = {0: 1.5, 1: 1.5, 2: 1.2, 3: 1.3, 4: 2.0, 5: 0.8}
    class_weight_op2 = {0: 3.0, 1: 1.0, 2: 1.5, 3: 1.5, 4: 2.5, 5: 0.8}
    class_weight_op3 = {0: 4.0, 1: 1.2, 2: 1.3, 3: 1.5, 4: 3.0, 5: 0.7}
    param_grid = {
        'n_estimators': [200, 300, 500, 700],
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [3, 5, 10],
        'min_samples_leaf': [3, 5],
        'max_features': ['sqrt', 'log2', 2],
        'class_weight': ['balanced', class_weight_op1, class_weight_op2, class_weight_op3]
    }
    recall_macro_scorer = make_scorer(recall_score, average='macro')
    modelRF = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=modelRF, param_distributions=param_grid, scoring=recall_macro_scorer, cv=5, error_score='raise')
    random_search.fit(X_train_scal_re, y_train_balanced)
    best_rf = RandomForestClassifier(**random_search.best_params_, random_state=42)
    best_rf.fit(X_train_scal_re, y_train_balanced)
    prediction_RF = best_rf.predict(X_test_scal)
    print("Random Forest Test Accuracy:", round(best_rf.score(X_test_scal, y_test), 4))
    print(classification_report(y_test, prediction_RF))
    print("Precision:", round(precision_score(y_test, prediction_RF, average='macro'), 4))
    print("Recall:", round(recall_score(y_test, prediction_RF, average='macro'), 4))
    with open(path_save + 'RF_trained_model.pkl', 'wb') as file:
        pickle.dump(best_rf, file)
    return best_rf

def train_xgboost(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save):
    param_grid_Xgb = {
        'booster': ['gbtree'],
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'objective': ["multi:softprob"],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.2, 1],
        'reg_lambda': [0, 1, 2],
        'num_class': [3]
    }
    import xgboost as xgb
    model_xgb = xgb.XGBClassifier(random_state=42)
    xgb_search = RandomizedSearchCV(estimator=model_xgb, param_distributions=param_grid_Xgb, cv=5, error_score='raise', n_iter=30)
    xgb_search.fit(X_train_scal_re, y_train_balanced)
    xgb_best = xgb.XGBClassifier(**xgb_search.best_params_, random_state=42)
    xgb_best.fit(X_train_scal_re, y_train_balanced)
    prediction_XGB = xgb_best.predict(X_test_scal)
    print("XGBoost Test Accuracy:", round(xgb_best.score(X_test_scal, y_test), 4))
    print(classification_report(y_test, prediction_XGB))
    print("Precision:", round(precision_score(y_test, prediction_XGB, average='macro'), 4))
    print("Recall:", round(recall_score(y_test, prediction_XGB, average='macro'), 4))
    with open(path_save + 'xgb_trained_model.pkl', 'wb') as file:
        pickle.dump(xgb_best, file)
    return xgb_best

def trainVoting_Classifier(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save):
    voting_clf = VotingClassifier(estimators=[
        ('RF', train_random_forest(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save)),
        ('XGB', train_xgboost(X_train_scal_re, y_train_balanced, X_test_scal, y_test, path_save))
    ], voting='soft')
    voting_clf.fit(X_train_scal_re, y_train_balanced)
    prediction_Voting_clf = voting_clf.predict(X_test_scal)
    print("Voting Classifier Test Accuracy:", round(voting_clf.score(X_test_scal, y_test), 4))
    print(classification_report(y_test, prediction_Voting_clf))
    print("Precision:", round(precision_score(y_test, prediction_Voting_clf, average='macro'), 4))
    print("Recall:", round(recall_score(y_test, prediction_Voting_clf, average='macro'), 4))
    with open(path_save + 'voting_trained_model.pkl', 'wb') as file:
        pickle.dump(voting_clf, file)
    return voting_clf