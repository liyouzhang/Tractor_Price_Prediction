from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

productgroup = ['ProductGroup_BL', 'ProductGroup_MG', 'ProductGroup_SSL',
       'ProductGroup_TEX', 'ProductGroup_TTT', 'ProductGroup_WL']

productsize = ['ProductSize_Compact', 'ProductSize_Large',
       'ProductSize_Large / Medium', 'ProductSize_Medium', 'ProductSize_Mini',
       'ProductSize_Small']

# auctioneerids = ['auctioneerID_0.0', 'auctioneerID_1.0', 'auctioneerID_2.0',
#        'auctioneerID_3.0', 'auctioneerID_4.0', 'auctioneerID_5.0',
#        'auctioneerID_6.0', 'auctioneerID_7.0', 'auctioneerID_8.0',
#        'auctioneerID_9.0', 'auctioneerID_10.0', 'auctioneerID_11.0',
#        'auctioneerID_12.0', 'auctioneerID_13.0', 'auctioneerID_14.0',
#        'auctioneerID_15.0', 'auctioneerID_16.0', 'auctioneerID_17.0',
#        'auctioneerID_18.0', 'auctioneerID_19.0', 'auctioneerID_20.0',
#        'auctioneerID_21.0', 'auctioneerID_22.0', 'auctioneerID_23.0',
#        'auctioneerID_24.0', 'auctioneerID_25.0', 'auctioneerID_26.0',
#        'auctioneerID_27.0', 'auctioneerID_28.0', 'auctioneerID_99.0']


def selectcolumns(X):
    """Only keep columns that we want to keep.
    """
    keep_cols = ['YearMade','Age','MachineHoursCurrentMeter']
    for i in productgroup:
        keep_cols.append(i)
    for i in productsize:
        keep_cols.append(i)
    # for i in auctioneerids:
    #     keep_cols.append(i)
    X = X[keep_cols]
    return X



def getdummies(X):

    selected_for_dummies = {
        'ProductGroup': productgroup,
        'ProductSize':productsize
    #    'auctioneerID':auctioneerids
        }
    for col in selected_for_dummies.keys():
        dummies = pd.get_dummies(X[col],prefix=col)
        X[dummies.columns] = dummies
    return X

def medianYear(X):
    median_year = X['YearMade'].median()
    X.loc[X[X['YearMade'] == 1000]['YearMade'].index,'YearMade'] = median_year
    return X

def calculateAge(X):
    X['saledate']=pd.to_datetime(X['saledate'])
    X['Age'] = X['saledate'].apply(lambda x: x.year) - X['YearMade']
    return X

def replaceNaN(X):
    """Replace NaNs
    """
    num_col_name = ['MachineHoursCurrentMeter']
    cat_col_name = ['auctioneerID']

    dict = {}
    num_median = X[num_col_name].median().values.flatten()
    cat_mod = X[cat_col_name].mode().values.flatten()
    for col_name, value in zip(cat_col_name, cat_mod):
        dict[col_name] = value
    for col_name, value in zip(num_col_name, num_median):
        dict[col_name] = value
    X.fillna(value=dict, inplace=True)
    return X


def feature_eng(X):
    X = replaceNaN(X)
    X = medianYear(X)
    X = calculateAge(X)
    X = getdummies(X)
    X = selectcolumns(X)
    return X




