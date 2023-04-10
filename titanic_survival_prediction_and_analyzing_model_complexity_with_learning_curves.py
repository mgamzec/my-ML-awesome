############################ Titanic Survival Prediction with Machine Learning ###########################

########Libraries and Utilities #########

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from helpers.eda import *
from helpers.data_prep import *

warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

############################ Data Preprocessing & Feature Engineering ###########################

def titanic_data_prep(dataframe):
    print("Data Preprocessing...")
    # 1. Feature Engineering
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # 2. Outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # 3. Missing Values
    dataframe.drop("CABIN", inplace=True, axis=1)
    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
                                axis=0)

    # 4. Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # 5. Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01)

    # 6. One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # 7. Standart Scaler
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

df_prep = titanic_data_prep(df)

y = df_prep["SURVIVED"]
X = df_prep.drop(["PASSENGERID", "SURVIVED"], axis=1)

#################### Analyzing Model Complexity with Learning Curves with Random Forest ##########################
rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [4, 7, 11, 15, None],
             "max_features": [5, 8, 10, 13, "auto"],
             "min_samples_split": [8, 13, 15, 20],
             "min_samples_leaf": [2, 4, 6, 8],
             "n_estimators": [100, 200, 250, 300, 400]}


rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_,
                               random_state=17).fit(X, y)

# Final Model's CV Error:
cv_results = cross_validate(rf_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8327590511860175
cv_results['test_f1'].mean()
#  0.7658206525533824
cv_results['test_roc_auc'].mean()
# 0.8648893698893699

######################### Feature Importance ########################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X, 30)

######################## Analyzing Model Complexity with Learning Curves #######################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=5, save=False):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    if save:
        plt.savefig('Complexity_Curve.png')


rf_val_params = [["max_depth", [3, 10, 12, 15, 17, 20, None]],
                 ["max_features", [3, 7, 8, 10, 15, 20, 30]],
                 ["min_samples_split", [2, 5, 8, 12, 20,30, 50]],
                 ["min_samples_leaf", [1, 3, 5, 8, 15]],
                 ["n_estimators", [5, 10, 20, 50, 75]]]

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_model.get_params()
