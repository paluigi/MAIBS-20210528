import pandas as pd
import os
import time
import pickle

pwd = os.getcwd()
ml_path = os.path.join(pwd, "model")
pred_path = os.path.join(pwd, "preds")
pred_file = os.path.join(pred_path, "y_pred.csv")

if not os.path.isdir(pred_path):
  os.makedirs(pred_path)


# Model 1
def testLiveData():
  df_test = pd.read_csv('https://creditrisk-cusvdt2goq-ez.a.run.app/live')
  with open(os.path.join(ml_path, 'ML.pickle'), 'rb') as handle:
    scaler,featureSel,knn_best = pickle.load(handle)

  ##########################################################################
  ## Remove non informative columns
  out = df_test[['id_USA', '_C-member_id_USA']]
  df_test.drop(['id_USA', '_C-member_id_USA'], axis=1, inplace=True)

  #####################################################################
  ## Feature encoding: for simplicity I am using Label Encoding on categorical variables
  for c in df_test.columns:
        if df_test[c].dtypes == 'object':
            df_test[c]=df_test[c].astype('category').cat.codes

  ###################################################################
  ## Standard Scaler
  x = pd.DataFrame(scaler.transform(df_test),columns=df_test.columns)

  ###################################################################
  #Feature selection based on mutual information
  x = featureSel.transform(x)

  y_pred = knn_best.predict(x)
  out.insert(2,'_C-default_ind_pred',y_pred)
  return out


# Model 2
def testLiveData2():
  df_test = pd.read_csv('https://creditrisk-cusvdt2goq-ez.a.run.app/live')
  with open(ml_path + 'ML.pickle', 'rb') as handle:
    scaler,featureSel,rfc_best,sub_grades = pickle.load(handle)

  ##########################################################################
  ## Remove non informative columns
  out = df_test[['id_USA', '_C-member_id_USA']]
  df_test.drop(['id_USA', '_C-member_id_USA'], axis=1, inplace=True)

  #########################################################
  ## Transform variables
  ## Delta between start of credit history and loan application
  df_test["_C-earliest_cr_line_USA"] = pd.to_datetime(df_test["_C-earliest_cr_line_USA"])
  df_test["_C-issue_d_USA"] = pd.to_datetime(df_test["_C-issue_d_USA"])
  df_test["time_delta"] = df_test["_C-issue_d_USA"] - df_test["_C-earliest_cr_line_USA"]
  ## All back to numeric
  df_test["_C-earliest_cr_line_USA"] = pd.to_numeric(df_test["_C-earliest_cr_line_USA"])
  df_test["_C-issue_d_USA"] = pd.to_numeric(df_test["_C-issue_d_USA"])
  df_test["time_delta"] = pd.to_numeric(df_test["time_delta"])
  
  ## Ordered category for the credit rating
  grade_cat = pd.Categorical(df_test["_C-sub_grade_USA"], categories=sub_grades, ordered=True)
  df_test["_C-sub_grade_USA"] = pd.Series(grade_cat)
  df_test["_C-sub_grade_USA"] = df_test["_C-sub_grade_USA"].cat.codes

  #####################################################################
  ## Feature encoding: for simplicity I am using Label Encoding on categorical variables
  for c in df_test.columns:
        if df_test[c].dtypes == 'object':
            df_test[c]=df_test[c].astype('category').cat.codes

  ###################################################################
  ## Standard Scaler
  x = pd.DataFrame(scaler.transform(df_test),columns=df_test.columns)

  ###################################################################
  #Feature selection based on mutual information
  x = featureSel.transform(x)

  y_pred = rfc_best.predict(x)
  out.insert(2,'_C-default_ind_pred',y_pred)
  return out


# Start inifinte loop
while True:
  start_time = time.time()
  # Chose model, according to file train_model.py
  #out = testLiveData()
  out = testLiveData2()
  if not os.path.isfile(pred_file):
    out.to_csv(pred_file, index=False)
  else:
    out.to_csv(pred_file, index=False, mode='a', header=False)
  #print(out)
  print('## Time spent for training in seconds: ' + str(int(time.time() - start_time)))
  time.sleep(4)