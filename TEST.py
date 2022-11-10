###This function is for testing the csv dimension.
import numpy as np
import pandas as pd
import os,csv
test_mar_1=np.load('A_1.npy', allow_pickle = True)
test_mar_2=np.load('B_1.npy', allow_pickle = True)
# print(test_mar_2)
data=np.load('energy.npy')
# print(np.shape(data))
data_new=data.transpose((1,0,2))
# print(np.shape(data_new))
###for the LOCF test and similar for the NOCB
test_1=np.array([[1,0,0],[0,1,0],[0,0,1]])
test_2=np.array([[0,0,0],[0,0,0],[0,0,0]])
###for the interpolation test
test_3=np.array([[1/2,0,0],[0,1/2,0],[0,0,1/2]])
def get_result(A,B,data,a,b,mode='MAR'):
    sample=data_new[:,a:a+3,b:b+3]
    index = np.random.choice(range(40)[9:17], 5,replace=False)
    front = sample[index+1, :]
    back = sample[index-1, :]
    origin = sample[index, :]
    result = []
    for i in range(5):
        if mode == 'MAR':
            mar_coefficient=A@back[i]@B 
            mse = mse_calculate(origin[i], mar_coefficient)
        elif mode == 'Interpolation':
            inter_coefficient=(front[i]@A+back[i]@B)/2
            mse = mse_calculate(origin[i], inter_coefficient)
        elif mode == 'LOCF':
            LOCF_coefficient=front[i]@A
            mse = mse_calculate(origin[i], LOCF_coefficient)
        elif mode =='NOCB':
            NOCB_coefficient=back[i]@B
            mse = mse_calculate(origin[i], NOCB_coefficient)
        result.append(mse)
    sum = np.sum(np.array(result))
    return result, sum
# print(np.shape(sample))
# calculate for the mse of two matrix
def mse_calculate(X,Y):
    res=np.array([[0,0,0],[0,0,0],[0,0,0]])
    a=X.shape[0]
    b=X.shape[1]
    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j]=X[i][j]-Y[i][j]
    res=res.reshape(1,a*b)
    res=np.sum(res**2)/(a*b)
    return res
print(mse_calculate(test_1,test_2))
# result,res = get_result(test_1, test_2, data_new, 12, 24, 'LOCF')
# result,res = get_result(test_2, test_1, data_new, 12, 24, 'NOCB')
# result,res = get_result(test_3, test_3, data_new, 12, 24, 'Interpolation')
result,res = get_result(test_mar_1, test_mar_2, data_new, 0, 0, 'MAR')
print(result)
print(res)

