import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

from feature_engineering import *

sets = []

def read_files(files):
    for file in files:
        sets.append(pd.read_csv(file))

def rmsle(y_true, y_pred):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    (hard-coding the clipping here as a dumb hack for the Pipeline)
    """
    y_pred_clipped = np.clip(y_pred, 4750, None)
    log_diff = np.log(y_true+1) - np.log(y_pred_clipped+1) 
    return np.sqrt(np.mean(log_diff**2))


def modeling():

    #creat X,y and X_test
    X = sets[0].drop('SalePrice',axis=1)
    y = sets[0]['SalePrice']
    X = feature_eng(X)
    X_test = sets[1]
    X_test = feature_eng(X_test)

    l = LinearRegression()
    # clf.fit(X.reset_index(),y)
    # price = clf.predict(X_test.reset_index())

    params = {'fit_intercept':[True,False]}
    rmsle_scorer = make_scorer(rmsle,greater_is_better=False)
    clf = GridSearchCV(l,param_grid=params,scoring=rmsle_scorer,cv=3,n_jobs=-1)
    clf.fit(X.reset_index(),y)
    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))
    price = clf.predict(X_test.reset_index())

    return price

def submission(price):
    submit = sets[1][['SalesID']]
    submit['SalePrice'] = price
    submit.to_csv('tractor_price_prediction.csv', index = False)


if __name__ == '__main__':
    print("$ READ FILES")
    read_files(['data/Train.zip', 'data/Test.zip'])

    print("$ MODELING")
    price = modeling()

    print("$ SUBMISSION")
    submission(price)
