import pandas as pd
import os
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import feature_selection, metrics
from sklearn.utils import resample

pwd = os.getcwd()
data_path = os.path.join(pwd, "data")
ml_path = os.path.join(pwd, "model")

if not os.path.isdir(data_path):
    os.makedirs(data_path)

if not os.path.isdir(ml_path):
    os.makedirs(ml_path)


# Get data into dataframe
def loadCollectedTrainingData():
    df_train = pd.DataFrame()
    ## Load all training CSV collected so far
    list_files = os.listdir(data_path)
    ## print('## Load CSV files')
    for f in list_files:
        if ".csv" in f:
            # print(f)
            df_tmp = pd.read_csv(os.path.join(data_path, f))
            df_train = df_train.append(df_tmp, ignore_index=True)

    ## As per API documentation I need to remove dupplicates
    df_train.drop_duplicates(inplace=True)
    print("## Removed duplicates, training set shape: " + str(df_train.shape))
    return df_train


# Model number one
def trainModelOnCollectedData(df_train, path_files_ml):
    #######################################################################
    ## Rebalancing the dataset since _C-default_ind = 1 has far less samples
    ## Separate majority and minority classes
    df_majority = df_train[df_train["_C-default_ind"] == 0]
    df_minority = df_train[df_train["_C-default_ind"] == 1]

    ## Under-sample majority class (_C-default_ind = 0)
    ratio_calsess = df_minority.shape[0] / df_majority.shape[0]
    num_samples = int(df_majority.shape[0] / 2.5)
    df_majority_downsampled = resample(df_majority, replace=True, n_samples=num_samples)

    print(
        "## Class balance ratio: "
        + str(df_minority.shape[0] / df_majority_downsampled.shape[0])
    )
    ## Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    print("## Balanced training set shape: " + str(df_balanced.shape))

    ##########################################################################
    ## Define y and remove non informative columns from X (df_balanced)
    y = df_balanced["_C-default_ind"]
    df_balanced.drop(
        ["id_USA", "_C-member_id_USA", "_C-default_ind"], axis=1, inplace=True
    )

    #####################################################################
    ## Feature encoding: for simplicity I am using Label Encoding on categorical variables
    for c in df_balanced.columns:
        if df_balanced[c].dtypes == "object":
            # print(c)
            df_balanced[c] = df_balanced[c].astype("category").cat.codes

    ###################################################################
    ## Standard Scaler
    scaler = preprocessing.StandardScaler().fit(df_balanced)
    X = pd.DataFrame(scaler.transform(df_balanced), columns=df_balanced.columns)

    ###################################################################
    # Feature selection based on mutual information
    featureSel = feature_selection.SelectKBest(
        feature_selection.mutual_info_classif, k=10
    ).fit(X, y)
    X = featureSel.transform(X)

    #################################################################
    ## Grid Search cross-validation for KNeighborsClassifier
    parameters = {"n_neighbors": [5, 9, 15]}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, scoring="f1_micro", cv=5)
    clf.fit(X, y)
    print(clf.cv_results_)
    K_best = parameters["n_neighbors"][clf.cv_results_["rank_test_score"][0] - 1]
    knn_best = KNeighborsClassifier(n_neighbors=K_best).fit(X, y)

    #################################################################
    ## Storing the trainied methods
    with open(path_files_ml + "ML.pickle", "wb") as handle:
        pickle.dump([scaler, featureSel, knn_best], handle)
    print("## KNN model trained")


# Model number 2
def trainModel2(df_train, path_files_ml):
    #######################################################################
    ## Rebalancing the dataset since _C-default_ind = 1 has far less samples
    ## Separate majority and minority classes
    df_majority = df_train[df_train["_C-default_ind"] == 0]
    df_minority = df_train[df_train["_C-default_ind"] == 1]

    ## Under-sample majority class (_C-default_ind = 0)
    ratio_calsess = df_minority.shape[0] / df_majority.shape[0]
    num_samples = int(df_majority.shape[0] / 2.5)
    df_majority_downsampled = resample(df_majority, replace=True, n_samples=num_samples)

    print(
        "## Class balance ratio: "
        + str(df_minority.shape[0] / df_majority_downsampled.shape[0])
    )
    ## Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    print("## Balanced training set shape: " + str(df_balanced.shape))

    ##########################################################################
    ## Define y and remove non informative columns from X (df_balanced)
    y = df_balanced["_C-default_ind"]
    df_balanced.drop(
        ["id_USA", "_C-member_id_USA", "_C-default_ind"], axis=1, inplace=True
    )

    #########################################################
    ## Transform variables
    ## Delta between start of credit history and loan application
    df_balanced["_C-earliest_cr_line_USA"] = pd.to_datetime(
        df_balanced["_C-earliest_cr_line_USA"]
    )
    df_balanced["_C-issue_d_USA"] = pd.to_datetime(df_balanced["_C-issue_d_USA"])
    df_balanced["time_delta"] = (
        df_balanced["_C-issue_d_USA"] - df_balanced["_C-earliest_cr_line_USA"]
    )
    ## All back to numeric
    df_balanced["_C-earliest_cr_line_USA"] = pd.to_numeric(
        df_balanced["_C-earliest_cr_line_USA"]
    )
    df_balanced["_C-issue_d_USA"] = pd.to_numeric(df_balanced["_C-issue_d_USA"])
    df_balanced["time_delta"] = pd.to_numeric(df_balanced["time_delta"])

    ## Ordered category for the credit rating
    sub_grades = df_balanced["_C-sub_grade_USA"].unique()
    sub_grades.sort()
    grade_cat = pd.Categorical(
        df_balanced["_C-sub_grade_USA"], categories=sub_grades, ordered=True
    )
    df_balanced["_C-sub_grade_USA"] = pd.Series(grade_cat)
    df_balanced["_C-sub_grade_USA"] = df_balanced["_C-sub_grade_USA"].cat.codes

    #####################################################################
    ## Feature encoding: for simplicity I am using Label Encoding on categorical variables
    for c in df_balanced.columns:
        if df_balanced[c].dtypes == "object":
            # print(c)
            df_balanced[c] = df_balanced[c].astype("category").cat.codes

    ###################################################################
    ## Robust Scaler
    scaler = preprocessing.RobustScaler().fit(df_balanced)
    X = pd.DataFrame(scaler.transform(df_balanced), columns=df_balanced.columns)

    ###################################################################
    # Feature selection based on mutual information
    featureSel = feature_selection.SelectKBest(
        feature_selection.mutual_info_classif, k=10
    ).fit(X, y)
    X = featureSel.transform(X)

    #################################################################
    ## Grid Search cross-validation for Random Forest Classifier
    parameters = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, parameters, scoring="f1_micro", cv=5)
    clf.fit(X, y)
    print(clf.cv_results_)
    n_est_best = clf.cv_results_["param_n_estimators"][
        clf.cv_results_["rank_test_score"][0] - 1
    ]
    max_d_best = clf.cv_results_["param_max_depth"][
        clf.cv_results_["rank_test_score"][0] - 1
    ]
    rfc_best = RandomForestClassifier(
        n_estimators=n_est_best, max_depth=max_d_best
    ).fit(X, y)

    #################################################################
    ## Storing the trainied methods
    with open(path_files_ml + "ML.pickle", "wb") as handle:
        pickle.dump([scaler, featureSel, rfc_best, sub_grades], handle)
    print("## Random Forest model trained")


# Start infinite loop
while True:
    start_time = time.time()
    df_train = loadCollectedTrainingData()
    # Choose which model deploy
    # Same choice to be mande in the predict.py script
    # trainModelOnCollectedData(df_train, ml_path)
    trainModel2(df_train, ml_path)
    print(
        "## Time spent for training in seconds: " + str(int(time.time() - start_time))
    )
    time.sleep(120)
