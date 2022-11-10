# MAR-model-impute
Use mar model to impute the financial report
All implement procedure
Install the released version of tensorTS

use Dataset.py to produce the data with/without missing value

main.R--the most important function,to transfer the data to R,and to estimate the mar model's correlation matrix.

MAR_proj_method is for LSE initialization

MAR_lse_method is for LSE method of matrix autoregressive model

MAR_plot is for time series plot and acf plot

MAR_prediction is for prediction(based on the coefficient matrix)

test_data_produce.py is to create the dataset we need,we save the npy and csv file for further implement

TEST.py is for test with interpolation and other imputation method.

I also attach the test dataset.You can change it.

