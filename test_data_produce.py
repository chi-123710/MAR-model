from  torch.utils.data import Dataset, DataLoader
import torch
from fancyimpute import IterativeSVD, SoftImpute,BiScaler,MatrixFactorization,IterativeImputer
import numpy as np
import DataClean as dc
import pickle
import json
from DataClean import DataHandler



def GetNumpyData():
    '''
    Save the data that has been visualized in python, the date of data is aligned.
    :return:
    '''
    handler = dc.DataHandler()
    pickle_in = open(handler.cleanPath + '/cleanData.pickle', 'rb')
    new_dict = pickle.load(pickle_in)
    for sector in new_dict.keys():
        handler.numpy(new_dict[sector], sector=sector,save=True)






def loadData(sector)->object:
    quarter_file_path='./Data/Training/'+sector+'_quarter_data.npy'
    annual_file_path='./Data/Training/'+sector+'_annual_data.npy'
    date_stamp=np.load('./Data/CleanData/dataDate.npy')
    quarter_data=np.load(quarter_file_path)
    annual_data=np.load(annual_file_path)
    print('The quarterly data start from %s to %s'%(date_stamp[0][0],date_stamp[0][1]))
    print('The yearly data start from %s to %s'%(date_stamp[1][0],date_stamp[1][1]))
    return quarter_data,annual_data







def getDataset(sector,method='soft',normalize=True,top=10,zero=False,incomplete=False):

    def extractFeature(data,annual,sigma):
        '''
        Reserve the col[attribute] that at least sigma number of company has this feature
        :param top: top n years data
        :param data: Input data
        :param annual: whether it is annual data
        :param sigma: lower bound of the company number to preserve the feaature
        :param incomplete: whether return the incomplete dataset
        :return:
        '''

        index_dic = json.load(open('./Data/CleanData/standard_index.json'))

        if annual:
            print('The feature number of annual data is %d'%(data.shape[1]))
            date_index=index_dic['a_index'][:top]
            feature=np.array(index_dic['a_feature'])
            nan_score=np.isnan(data)
            nan_index=np.sum(nan_score,axis=0)<data.shape[0]*(1-sigma)

            data=data[:,nan_index]
            zero_score = (data == 0)
            zero_index = np.sum(zero_score, axis=0) < data.shape[0] * (1 - sigma)
            print('The feature number of annual data after filtering is %d' % (data[:, zero_index].shape[1]))
            return data[:,zero_index]
        else:
            print('The feature number of quarterly data is %d'%(data.shape[1]))
            date_index = index_dic['q_index'][:top]
            feature = np.array(index_dic['q_feature'])
            # for nan
            nan_score = np.isnan(data)
            nan_index = np.sum(nan_score, axis=0) < data.shape[0] * (1 - sigma)
            data=data[:,nan_index]
            # for zero
            zero_score=(data==0)
            zero_index=np.sum(zero_score, axis=0) < data.shape[0] * (1 - sigma)
            print('The feature number of quarterly data after filtering is %d' % (data[:, zero_index].shape[1]))
            return data[:,zero_index]


    def delCompany(data,sigma,anuual):
        '''
           Del the row [company] that havs too many missing value
        :param data:
        :param sigma:
        :param anuual:
        :return:
        '''

        if not anuual:
            nan_score=np.sum(np.isnan(data),axis=1)
            nan_index = nan_score < data.shape[1] * (1 - sigma)
            company_index=np.unique(np.array(np.where(nan_index==False))//(top*4)) # this is list of company number to drop in the future
            reserve_index=np.ones(data.shape[0])
            first=np.array(np.arange(company_index[0]*top*4,company_index[0]*top*4+top*4))

            for index in company_index[1:]:
                first=np.vstack([first,np.array(np.arange(index*top*4,index*top*4+top*4))])

            drop_index=first.flatten()
            reserve_index[drop_index]=0
            reserve_index=reserve_index.astype(np.bool_)
            number_of_company=company_index.shape[0]
            data=data[reserve_index]

            return data,number_of_company

        else:
            nan_score = np.sum(np.isnan(data), axis=1)
            nan_index = nan_score < data.shape[1] * (1 - sigma)
            company_index = np.unique(
                np.array(np.where(nan_index == False)) // top)  # this is list of company number to drop in the future
            reserve_index = np.ones(data.shape[0])
            first = np.array(np.arange(company_index[0] * top , company_index[0] * top  + top ))

            for index in company_index[1:]:
                first = np.vstack([first, np.array(np.arange(index * top , index * top  + top))])

            drop_index = first.flatten()
            reserve_index[drop_index] = 0
            reserve_index = reserve_index.astype(np.bool_)
            number_of_company = company_index.shape[0]
            data = data[reserve_index]

            return data,number_of_company


    quarter_data,annual_data=loadData(sector)
    # handler=DataHandler()
    # data = pickle.load(open(handler.cleanPath + '/cleanData.pickle', 'rb'))['Tech']
    quarter_dim=quarter_data[:,:top*4,:].shape
    annual_dim=annual_data[:,:top,:].shape


    quarter_data=quarter_data[:,:top*4,:].reshape(-1,quarter_data.shape[2])
    annual_data=annual_data[:,:top,:].reshape(-1,annual_data.shape[2])
    # reshape to filter the data
    pass
    # extract
    incomplete_annual_data=extractFeature(annual_data,annual=True,sigma=0.8)
    incomplete_quarter_data=extractFeature(quarter_data,annual=False,sigma=0.8)



    incomplete_annual_data, annual_company_num=delCompany(incomplete_annual_data,anuual=True,sigma=0.8)
    incomplete_quarter_data,quarter_company_num=delCompany(incomplete_quarter_data,anuual=False,sigma=0.8)



    if incomplete: # after del company
        return torch.as_tensor(incomplete_quarter_data.reshape(quarter_dim[0]-quarter_company_num, quarter_dim[1], -1)).float(), \
               torch.as_tensor(incomplete_annual_data.reshape(annual_dim[0]-annual_company_num, annual_dim[1], -1)).float()


    # handler=DataHandler()
    # data = pickle.load(open(handler.cleanPath + '/cleanData.pickle', 'rb'))['Tech']




    if zero:
        incomplete_quarter_data[np.isnan(incomplete_quarter_data)]=0
        incomplete_annual_data[np.isnan(incomplete_annual_data)]=0
        return torch.as_tensor(incomplete_quarter_data.reshape(quarter_dim[0]-quarter_company_num, quarter_dim[1], -1)).float(), \
               torch.as_tensor(incomplete_annual_data.reshape(annual_dim[0]-annual_company_num, annual_dim[1], -1)).float()




    complete_quarter_data, complete_annual_data = None, None

    if method=='soft':
        complete_quarter_data = SoftImpute().fit_transform(incomplete_quarter_data)
        complete_annual_data = SoftImpute().fit_transform(incomplete_annual_data)
    elif method=='SVD':
        complete_quarter_data = IterativeSVD().fit_transform(incomplete_quarter_data)
        complete_annual_data = IterativeSVD().fit_transform(incomplete_annual_data)
    elif method=='iter':
        complete_quarter_data = IterativeImputer().fit_transform(incomplete_quarter_data)
        complete_annual_data = IterativeImputer().fit_transform(incomplete_annual_data)


    complete_annual_data=complete_annual_data.reshape(annual_dim[0]-annual_company_num,annual_dim[1],-1)
    complete_quarter_data = complete_quarter_data.reshape(quarter_dim[0]-quarter_company_num, quarter_dim[1], -1)

    handler = DataHandler()
    data = pickle.load(open(handler.cleanPath + '/cleanData.pickle', 'rb'))['Tech']


    # 后normalize
    if normalize:
        complete_annual_data = dataNormalization(complete_annual_data,method='std')
        complete_quarter_data = dataNormalization(complete_quarter_data,method='std')



    return torch.as_tensor(complete_quarter_data).float(), \
           torch.as_tensor(complete_annual_data).float()



def dataNormalization(data,method='std'):

    if method=='std':
        data_mean=np.nanmean(data,axis=0)
        data_std=np.nanstd(data,axis=0)
        data=(data-data_mean)/data_std
        return data
    else:
        data_max=np.nanmax(data,axis=0)
        data_min=np.nanmin(data,axis=0)
        data_mean=np.nanmean(data,axis=0)
        data=(data)/(data_max)
        return data





def creatData2Label(data_len=3,company_num=5,quarter=True,sector='Tech',type='raw'):
    '''
    Slice the whole dataset into several patch to find the local pattern
    :param data_len:
    :param company_num:
    :return:
    '''

    # call the getDateset with different setting

    if type=='raw':
        quarter_data, annual_data = getDataset(sector, incomplete=True)
    elif type=='zero':
        quarter_data, annual_data = getDataset(sector, incomplete=False,zero=True)
    else:
        quarter_data, annual_data = getDataset(sector)


    # Data get the three dim, first one is company num, second one is the date,
    # and last one is feature
    data_list=[]
    label_list=[]
    print(quarter_data.shape)
    print(annual_data.shape)


    if quarter:
        for company in range(quarter_data.shape[0] - company_num):
            for i in range(quarter_data.shape[1] - data_len):
                data = quarter_data[company:company + company_num, i:i + data_len, :]
                label = quarter_data[company:company + company_num:, i + data_len, :]
                data_list.append(data)
                label_list.append(label)
        return data_list,label_list

    else:
        for company in range(annual_data.shape[0] - company_num):
            for i in range(annual_data.shape[1] - data_len):
                data_list.append(annual_data[company:company + company_num, i:i + data_len, :])
                label_list.append(annual_data[company:company + company_num:, i + data_len, :])

        return data_list, label_list




class PredictDataset(Dataset):

        def __init__(self,type='raw',sector='Tech',time_len=3,company_num=5,quarter=True):
            data_list,label_list=creatData2Label(data_len=time_len,company_num=company_num,quarter=quarter,type=type,sector=sector)
            assert len(data_list)==len(label_list),'Please check the length of data and label, not equal'
            self.data_list=data_list
            self.label_list=label_list

        # For LSTM
        def __getitem__(self, idx):
            data = (self.data_list[idx])
            data=data.permute(1,0,2).contiguous().view(3,-1)
            label = (self.label_list[idx])
            label = label.flatten()
            label = label.unsqueeze(0)
            return data, label


        # for MLP
        # def __getitem__(self, idx):
        #     data = torch.tensor((self.data_list[idx])).flatten()
        #     label = torch.tensor((self.label_list[idx])).flatten()
        #
        #     return data, label

        def __len__(self):
            return len(self.label_list)  # 返回数据的数量


class ImputeDataset(Dataset):
    def __init__(self, sector='Tech', quarter=True):
        data_list = getIncompleteData(sector=sector)
        self.data_list = data_list

    def __getitem__(self, idx):
        data = torch.Tensor((self.data_list[idx]).flatten())
        return data

    def __len__(self):
        return len(self.data_list)  # 返回数据的数量


def getIncompleteData(sector):

    quarterly_data_list = []


    quarter_data, annual_data = getDataset(sector=sector,incomplete=True)

    for company in range(quarter_data.shape[0]):
        for quarter in range(quarter_data.shape[1]):
            quarterly_data_list.append(quarter_data[company,quarter,:])

    print(torch.sum(torch.isnan(quarter_data)))
    return quarterly_data_list




def main():

    handler=DataHandler()
    q_data,a_data=getDataset(sector='Estate',incomplete=False)
    
    print(q_data.shape)
    import os,csv

    q_data,a_data=q_data.numpy(),a_data.numpy()
    q_data=dataNormalization(q_data,method='std')
    np.save('estate.npy',q_data)
    import csv

    # python2可以用file替代open

    with open("estate.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(q_data)

    # data = pickle.load(open(handler.cleanPath + '/cleanData.pickle', 'rb'))['Tech']
    #
    #
    # q,a=loadData('Tech')
    # print(q,a)

if __name__=='__main__':
    main()