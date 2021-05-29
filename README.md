# MAIBS-20210528
Exercise for Machine Learning

This exercise want to be a real world challenge where we need to analyse customer’s data in order to estimate the associated credit risk. In specific, a USA based financial institution is sharing its historical data via API and is asking to training a ML model able to anticipate whatever a new customer could be prone to default (_C-default_ind = 1).

The API is available here https://creditrisk-cusvdt2goq-ez.a.run.app/ and offers 3 methods:

    A static training set with historical data
    Sequences of updates with new training samples available every 60 seconds
    Every 5 seconds provides new customer’s data (testing set) that need to be evaluated.

As output we need to create a CSV file with the following header where _C-default_ind_pred is 1 when a customer is likely to default.

['id_USA', '_C-member_id_USA', '_C-default_ind_pred']

To address this challenge the IT department already developed 3 services for collecting, processing and estimating the default. To obtain the expected results all 3 services need to run simultaneously

    [A] EX_1_CollectTrainingData.ipynb – This service collects historical data and listens to updates. The collced data is stored in Google Drive. https://colab.research.google.com/drive/1JWm0E7a_VgAsyt-ZJJcN1tTEUQ7BbY4A#scrollTo=VRmdjZUYjh0d

    [B] EX_1_ModelTraining.ipynb – This service is a draft pipeline where a ML model is trained and the trained objects are stored as pickle on Google Drive. https://colab.research.google.com/drive/11_5TtpfGO6mq193d3oPPOubIB_peIJHa#scrollTo=aF38zihQeFVh

    [C] EX_1_TestLiveData.ipynb – Here the trained model is used to predict the default and generate the output CSV https://colab.research.google.com/drive/10e9zLhIjepIsAeXuuPxtfsxm_mxml1hg#scrollTo=Yu9MJxKWOp0w

Now, this USA based financial institution is haring you as data scientist to improve the performances. (Please note that in USA the average salary for a senior data scientist is around $120K/year) Using Colab or your PC start working on the dataset and enhance both [B] EX_1_ModelTraining.ipynb and [C] EX_1_TestLiveData.ipynb
