import numpy as np
import pandas as pd
import json
import os
from DealDF import dealDF
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


start_year = '1970'
end_year = '2021'
start_quarter = '1970-1'
end_quarter = '2022-1'

class DataHandler:
    '''
    Handler Class to deal with the Data from DataDownloader
    '''

    def __init__(self):
        '''
        Initializing the basic field and path to save
        '''
        self.dataPath = './Data/Fundamental/US'
        self.symbolList = './Data/SymbolList/US.npy'
        self.availablePath = './Data/SymbolList' + '/available_list.npy'
        self.cleanPath = './Data/CleanData'
        self.dropList_q = []
        self.dropList_a = []

    def getSymbolList(self):
        '''
        Get the symbol list from the saved file
        :return: numpy format symbollist
        '''
        return np.load(self.symbolList)

    def getFileName(self):
        '''
        Get the list of filename
        :return: return the list of filename
        '''
        file_list = [filename for filename in os.listdir(self.dataPath) if filename.endswith('json')]
        return file_list

    def getCap(self):
        '''
        Get the Market Cap of each company and filter some company financial report which is None, Dumps those
        :return: save the file to local file
        '''

        stock_basis = []
        file_list = self.getFileName()

        count = 0

        for file in file_list:

            file_path = os.path.join(self.dataPath, file)

            try:
                general_data = (json.load(open(file_path)))['General']
                highlight = ((json.load(open(file_path)))['Highlights'])
                code = general_data['Code']
                cap = highlight['MarketCapitalizationMln']
                sector = general_data['Sector']
                industry = general_data['Industry']

                if cap and sector and industry:
                    info = code, cap, sector, industry  # The unit of  Cap is Million
                    stock_basis.append(info)

            except Exception as e:
                print(count)
            finally:
                count += 1

        with open(self.availablePath, 'w') as f:
            np.save(self.availablePath, stock_basis)

    def getSectorList(self):
        '''
        Get the company's sector list and return it combined with the available company list
        :return:
        '''
        if os.path.exists(self.availablePath):
            return (list(set(np.load(self.availablePath)[:, 2]))), np.load(self.availablePath)
        else:
            raise FileNotFoundError

    def getTop(self, top=100, getList=False):
        '''
        Get the top N MarketCap company's symbol in each Sector
        The sector is categorizes within this function by hard code directly
        :param top: top n
        :param getList: only get the Sector list
        :return:  Dic of sector that contains all the symbol financial report within that sector/Industry
        '''
        sector_list, stock = self.getSectorList()
        data = pd.DataFrame(stock, columns=['Code', 'Cap', 'Sector', 'Industry'])
        data_by_sector = (data.groupby(['Sector']))

        # Hard-coded categories

        sector_merge_dict = {'Financial': ['Banks', 'Capital', 'Markets', 'Financial'],
                             'Consumer': ['Consumer'],
                             'Industrial': ['Industrial'],
                             'Tech': ['Technology'],
                             'Healthcare': ['Health', 'Pharmaceuticals'],
                             'Utilities': ['Utilities'],
                             'Energy': ['Energy'],
                             'Materials': ['Basic Materials', 'Materials'],
                             'Communication': ['Communication'],
                             'Estate': ['Estate']}

        if getList:
            return list(sector_merge_dict.keys())

        sector_data = {}

        for key in sector_merge_dict.keys():
            sector_data[key] = []

        for name, data in data_by_sector:
            find = False
            for key in sector_merge_dict.keys():
                if find:
                    break
                for indus in sector_merge_dict[key]:
                    if indus in name:
                        find = True
                        # print(str(name) + ' -----> ' + key,end='\n\n')
                        sector_data[key].append(data)
                    if find:
                        break
            if not find:
                # print('No corresponding sector for %s' % (name))
                pass

        for key in sector_data.keys():

            if len(sector_data[key]) == 1:
                sector_data[key] = sector_data[key][0]
                continue
            sector_data[key] = pd.concat(sector_data[key], axis=0)

        for key, value in sector_data.items():
            # 一开始这里value和 data[key]指向同一个

            value['Cap'] = value['Cap'].astype('float')
            sector_data[key] = value.sort_values(by=['Cap'], ascending=False)

            # 修改过后内存地址就不是同一个了

        for key, value in sector_data.items():
            if value.shape[0] >= 100:
                sector_data[key] = value.head(top)
                sector_data[key] = sector_data[key]['Code'].tolist()
            else:
                raise Exception('Please check the data source')
        print(1)
        return sector_data

    def dropDupilcatedCol(self, data):
        '''
        Since we merge the data, therefore drop the duplicated feature by exploring their postfix
        :param data: the data that need to remove the duplicated feature (Dataframe)
        :return: non-duplicated feature data
        '''

        to_drop_list = [col for col in data if col.endswith('_y')]
        # print('We will drop {} columns due to the duplication'.format(len(to_drop_list)),end='\n\n')
        data.drop(to_drop_list, axis='columns', inplace=True)
        for c in data.columns:
            if c.endswith('_x'):

                new_col = c[:len(c) - 2]

                # if new col name has been taken,drop the one with _x
                if new_col in data:
                    # print(f'{new_col} is already in the data\'s columns, we\' gonna drop it',end='\n\n')
                    data.drop(c, axis='columns', inplace=True)
                # new col name for data, replace without _x
                else:
                    data.rename(columns=({c: new_col}), inplace=True)
                    # print(f'replace the {c} with {new_col}',end='\n\n')

        return data

    def yearDate(self, data, start, end):
        '''
        Fix the financial report's time slot, fill the missing date with all nan features
        :param data: Data need to be tackled (Dataframe)
        :param start: annual xxxx
        :param end: ditto
        :return: The data that have been processed with give start date and end date
        '''

        data['date'] = data['date'].apply(str)  # in pandas str will be shown as object tyoe

        assert int(start) >= 1980 and int(end) <= 2021 and int(start) <= int(end), 'Invalid Year para'

        date_list = []

        if start == end:
            date_list = [start]

        else:
            for year in range(int(start), int(end) + 1):
                date_list.append(str(year))

        date_list = date_list[::-1]

        datetimeIndex = pd.Index(date_list, name='date')

        data = data.set_index("date").reindex(datetimeIndex).reset_index()

        return data

    def quarterDate(self, data, start, end, require=False):
        '''
        Filter the data according to give time slot
        :param data: data (Dataframe)
        :param start: start date year-quarter / quarter xxxx-x
        :param end:  ditto
        :return: The data that have been processed with give start date and end date
        '''

        if data.index.duplicated().any():  # return a index with same len as index is

            print(data[data.index.duplicated()])  # manually check the duplication

        data['date'] = data['date'].apply(str)

        assert int(end[-1]) <= 4 and int(end[-1]) >= 1, 'Quarter should be one of [1,2,3,4]'

        assert int(end[:4]) <= 2022 and int(end[:4]) >= 1980 and int(start[:4]) <= int(
            end[:4]), 'Exceed available data range'

        if int(end[:4]) == 2022:
            assert int(end[-1]) == 1, 'The 2022 Year only has quarter one report'

        date_list = []

        if start[:4] != end[:4]:

            for quarter in range(int(start[-1]), 5):
                date_list.append(start[:4] + '-' + str(quarter))

            for year in range(int(start[:4]) + 1, int(end[:4])):
                for quarter in ['-1', '-2', '-3', '-4']:
                    date_list.append(str(year) + quarter)

            for quarter in range(int(end[-1])):
                date_list.append(end[:4] + '-' + str(quarter + 1))

        else:
            for quarter in range(int(end[-1])):
                date_list.append(end[:4] + '-' + str(quarter + 1))

        date_list = date_list[::-1]

        row_num = len(date_list)

        datetimeIndex = pd.Index(date_list, name='date')

        data = data.set_index('date')

        data = data.reindex(datetimeIndex).reset_index()

        if require:
            return row_num

        return data

    def dropDuplicatedRow(self, df):
        '''
        Reference: # https://stackoverflow.com/questions/54406781/dropping-duplicate-observations-with-more-missing-values
        Remove the duplicated row with same report date, selecting the row with fewest missing value and drop all others duplicated one
        :param df: the data that need to be dealt with (Dataframe)
        :return: filtered data (Dataframe)
        '''

        x = df.loc[df.isnull().sum(axis=1).groupby(df.date).idxmin()]

        if df['date'].str.contains('-').any():
            x[['year', 'quarter']] = x['date'].str.split('-', expand=True)
            data = x.sort_values(by=['date', 'quarter'], ascending=[False, False])
            data.drop(['year', 'quarter'], axis='columns', inplace=True)
            return data

        data = x.sort_values(by='date', key=lambda x: x.astype('int64'), ascending=False)

        return data

    def saveStockData(self, q_start, q_end, a_start, a_end, save=False, prune=False,get_company_list=False):
        '''

        :param q_start: quarterly report start date xxxx-x string
        :param q_end:  quarterly report end date xxxx-x string
        :param a_start: ditto
        :param a_end: ditto
        :param save: whether save the data boolean
        :param prune: whether pre-pruned the data boolean
        :return: the processed data Dictionary
        '''

        def dataFliter(data):
            '''
            Basically convert all convertible column to numerical data, otherwise NaN
            :param data: Dataframe
            :return:  The
            '''

            annual = False

            if not data['date'].str.contains('-').any():  # Since the naming rule, add - to drop later
                annual = True
                data['date'] = data['date'].apply(lambda x: x + '-')

            data.set_index('date', inplace=True)

            for c in data.columns:
                # NoneType also convert to NaN
                data[c] = pd.to_numeric(data[c], errors='coerce')

            dropList = list(pd.isnull(data).all())

            if annual:
                if sum(dropList) > sum(self.dropList_a):
                    print('The num of completely missing col is %s' % (sum(dropList)))
                    self.dropList_a = dropList

            else:
                if sum(dropList) > sum(self.dropList_q):
                    print('The num of completely missing col is %s' % (sum(dropList)))
                    self.dropList_q = dropList

            return data

        def validateData(data):
            '''
            Validating the data integrity, specifically check the null dic value with in each finanical report [Earning, Income, Balance, Cash]
            :param data: Dataframe
            :return: Boolean whether the input data is valid
            '''
            if data['Earnings']['History'] == {} or data['Financials']['Income_Statement']['yearly'] == {} \
                    or data['Financials']['Balance_Sheet']['yearly'] == {} or data['Financials']['Cash_Flow'][
                'yearly'] == {}:
                # print('This is some problem with this json file,the company name is {}'.format(data['General']['Code']))
                return False
            else:
                return True

        def compute_quarter(start, end):
            return (int(end[:4]) - int(start[:4])) * 4 + int(start[-1]) - int(end[-1])

        def compute_year(start, end):
            return int(end[:4]) - int(start[:4])

        sector_symbol_list = self.getTop()

        financial_data = dict()

        for key in sector_symbol_list.keys():
            financial_data[key] = {}
            for symbol in sector_symbol_list[key]:
                financial_data[key][symbol] = {'quarter': None, 'annual': None}

        total = sum([len(symbol_list) for _, symbol_list in financial_data.items()])
        count = 0

        get_index = True


        existing_dic = {'year': (start_year, end_year), 'quarter': (start_quarter, end_quarter)}

        for sector, symbol_list in financial_data.items():

            count_annual_existing = np.zeros(shape=[compute_year(start_year, end_year)+1, 1])  # start from 1970 to 2022

            count_quarter_existing = np.zeros(
                shape=[compute_quarter(start_quarter, end_quarter)+1 , 1])  # strat from 1970-1-2022-1  1970-1---2022-1 208+1=209

            for symbol in symbol_list:

                if count % 10 == 0:
                    print(f'The Progress is {100 * count / total}%')

                file_name = self.dataPath + '/' + symbol + '.json'
                symbol_data = json.load(open(file_name))

                if validateData(symbol_data):

                    quarter_data, annual_data = dealDF(symbol_data)  # Call the complex function

                    annual_data = self.dropDuplicatedRow(self.dropDupilcatedCol(annual_data))
                    quarter_data = self.dropDuplicatedRow(self.dropDupilcatedCol(quarter_data))

                    # fill the year existing companies

                    the_first_date = (annual_data.iloc[len(annual_data) - 1, :]['date'])

                    no_exist = compute_year(start_year, the_first_date) # e.g. 1980 first year 1970 then first 10 year no data, the first one should be index [10]
                    count_annual_existing[:len(count_annual_existing) - no_exist, :] += 1 # then the last 10 should be zero, 52 years-10yesrs = 42 前42有1 这里+1 涵盖 42 year



                    the_first_date = (quarter_data.iloc[len(quarter_data) - 1, :]['date'])

                    no_exist = compute_quarter(start_quarter, the_first_date)

                    count_quarter_existing[:len(count_quarter_existing) - no_exist, :] += 1


                    # Check the dual duplicated columns
                    xlist = []
                    ylist = []
                    for c in quarter_data.columns:
                        if c.endswith('_x'):
                            xlist.append(c)
                        if c.endswith('_y'):
                            ylist.append(c)

                    assert len(xlist) == len(ylist)

                    assert quarter_data['date'].duplicated().any() == False and annual_data[
                        'date'].duplicated().any() == False, 'Duplicated Index Found'

                    annual_data = dataFliter(self.yearDate(annual_data, a_start, a_end))
                    quarter_data = dataFliter(self.quarterDate(quarter_data, q_start, q_end))

                    financial_data[sector][symbol]['quarter'] = quarter_data
                    financial_data[sector][symbol]['annual'] = annual_data

                    if get_index:
                        quarter_index = quarter_data.index.tolist()
                        quarter_feature = quarter_data.columns.tolist()
                        annual_index = annual_data.index.tolist()
                        annual_feature = annual_data.columns.tolist()

                        standard_dic = {'q_index': quarter_index,
                                        'q_feature': quarter_feature,
                                        'a_index': annual_index,
                                        'a_feature': annual_feature}
                        with open(self.cleanPath + '/standard_index.json', 'w') as f:
                            f.write(json.dumps(standard_dic))
                        get_index = False

                    count += 1

            existing_dic[sector] = (count_quarter_existing, count_annual_existing)



        self.dropList_a = [not val for val in self.dropList_a]
        self.dropList_q = [not val for val in self.dropList_q]

        del_list = []
        for sector, sector_data in financial_data.items():
            for symbol, f_data in sector_data.items():
                for type, report in f_data.items():

                    if report is None:
                        # print(f'Gonna del the data of {symbol}, since it is None')
                        del_list.append((sector, symbol))
                        break

        for item in del_list:
            del financial_data[item[0]][item[1]]

        if prune:
            for sector, sector_data in financial_data.items():
                for symbol, f_data in sector_data.items():
                    for type, report in f_data.items():

                        if type == 'quarter':
                            financial_data[sector][symbol][type] = report.loc[:, self.dropList_q]
                        else:
                            financial_data[sector][symbol][type] = report.loc[:, self.dropList_a]

        if not os.path.exists(self.cleanPath):
            os.makedirs(self.cleanPath)

        if get_company_list:

            company_dict=dict()

            for sector,sector_data in financial_data.items():
                company_dict[sector]=list(sector_data.keys())
            np.save(self.cleanPath+'/Sector2Company.npy',company_dict)
            print('The company list of each sector has been saved')


        if save and prune:
            pickle_out = open(self.cleanPath + '/lightCleanData.pickle', 'wb')
            pickle.dump(financial_data, pickle_out)

            np.save(self.cleanPath + '/Existing_num.npy', existing_dic)

            print('File size pickle file is',
                  round(os.path.getsize(self.cleanPath + '/cleanData.pickle') / (1024 ** 2), 1), 'MB')
            print('The data has been saved')

        elif save and not prune:
            pickle_out = open(self.cleanPath + '/cleanData.pickle', 'wb')
            pickle.dump(financial_data, pickle_out)

            np.save(self.cleanPath + '/Existing_num.npy', existing_dic)

            print('File size pickle file is',
                  round(os.path.getsize(self.cleanPath + '/cleanData.pickle') / (1024 ** 2), 1), 'MB')

            print('The data has been saved')

        np.save(self.cleanPath+'/dataDate.npy',[(q_start,q_end),(a_start,a_end)])


        return financial_data

    def numpy(self, sector_data, sector, count_only=False,save=False):
        '''
        Convert the Dataframe into numpy array
        :param sector_data: The data within one specified sector
        :param count_only: Whether only return the statistical result of missing value
        :param save: whether save the numpy data
        :return: converted numpy data/statistical result of missing value
        '''

        quarter_dim = list(iter(sector_data.values()).__next__().values())[0].shape
        annual_dim = list(iter(sector_data.values()).__next__().values())[1].shape

        company_num = len(sector_data)

        quarter_np_data = np.zeros(shape=(company_num, quarter_dim[0], quarter_dim[1]), dtype='object')
        annual_np_data = np.zeros(shape=(company_num, annual_dim[0], annual_dim[1]), dtype='object')

        for num, (symbol, fina_data) in enumerate(sector_data.items()):
            for type, report in fina_data.items():
                if type == 'quarter':
                    quarter_np_data[num, :, :] = report.to_numpy()
                elif type == 'annual':
                    annual_np_data[num, :, :] = report.to_numpy()

        quarter_np_data = quarter_np_data.astype(np.float64)
        annual_np_data = annual_np_data.astype(np.float64)

        quarter_logistics = (~(np.isnan(quarter_np_data))).astype(np.float64)
        annual_logistics = (~(np.isnan(annual_np_data))).astype(np.float64)

        annual_count_nonmissing = np.sum(annual_logistics, axis=0)
        quarter_count_nonmissing = np.sum(quarter_logistics, axis=0)

        if save:

            if not os.path.exists('./Data/Training'):
                os.mkdir('./Data/Training')

            np.save('./Data/Training/' + sector + '_quarter_data.npy', quarter_np_data)
            np.save('./Data/Training/' + sector + '_annual_data.npy', annual_np_data)
            print('The data for %s has been saved' % (sector))
            return None


        if count_only:
            return quarter_count_nonmissing, annual_count_nonmissing, company_num


        return quarter_np_data, annual_np_data

    def viusalizeMissingData(self, X, sector):
        '''
        Visualize the data of a company within a given sector
        :param X: Ether yearly/quarterly report of the company
        :param sector: sector of company
        :return: None
        '''
        cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
        index_dic = json.load(open(self.cleanPath + '/standard_index.json'))
        quarter_index = index_dic['q_index']
        quarter_feature = index_dic['q_feature']
        annual_index = index_dic['a_index']
        annual_feature = index_dic['a_feature']

        if X.shape[0] == len(annual_index):
            x_label = annual_feature
            y_label = annual_index
            fig, ax = plt.subplots(figsize=(20, 15))
            ax = sns.heatmap(X, xticklabels=x_label, yticklabels=y_label, vmin=0, vmax=1, cmap=cmap, square=True,
                             cbar_kws={"shrink": .6})
            ax.set_title("Heatmap for missing value of %s industry" % (sector))
            plt.tight_layout()

        else:
            x_label = quarter_feature
            y_label = quarter_index
            fig, ax = plt.subplots(figsize=(30, 15))
            ax = sns.heatmap(X.T, xticklabels=y_label, yticklabels=x_label, vmin=0, vmax=1, cmap=cmap, square=True,
                             cbar_kws={"shrink": .3})
            ax.set_title("Heatmap for missing value of %s industry" % (sector))
            plt.tight_layout()




        # # fig.set_size_inches(12, 12)
        # # im = ax.imshow(heatmap, cmap=map_vir)
        
        # # ax.set_xticks(np.arange(len(x_label)))
        # # ax.set_yticks(np.arange(len(y_label)))
        
        # # ax.set_xticklabels(x_label)
        # # ax.set_yticklabels(y_label)
        
        
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")
        
        # for i in range(len(y_label)):
        #     for j in range(len(x_label)):
        #         text = ax.text(j, i, X[i, j],
        #                        ha="othercenter", va="center", color="red", fontsize=9)
        # ax.set_title("Heatmap for missing value of %s industry" % (sector))
        # fig.tight_layout()
        # plt.colorbar(im)
        plt.show()


def main():
    np.seterr(divide='ignore', invalid='ignore')

    def compute_quarter(start, end):
        return (int(end[:4]) - int(start[:4])) * 4 - int(start[-1]) + int(end[-1])

    def compute_year(start, end):
        return int(end[:4]) - int(start[:4])

    handler = DataHandler()

    save = input('whether save the new Data [yes or others]:')

    if save == 'yes':

        q_start = input('Enter quarterly report start date like xxxx-x:')
        q_end = input('Enter quarterly report end date like xxxx-x:')

        a_start = input('Enter yearly report start date like xxxx:')
        a_end = input('Enter yearly report end date like xxxx:')

        prune = input('wehter pre-prune the data by all nan columns [yes or no]:')


        if prune == 'yes':
            handler.saveStockData(q_start=q_start, q_end=q_end, a_start=a_start, a_end=a_end, save=True, prune=True)
        else:
            handler.saveStockData(q_start=q_start, q_end=q_end, a_start=a_start, a_end=a_end, save=True)

    option = []

    while 1:

        dataDate=np.load(handler.cleanPath+'/dataDate.npy',allow_pickle=True)

        q_start,q_end,a_start,a_end=dataDate[0][0],dataDate[0][1],dataDate[1][0],dataDate[1][1]

        print('The start date of quarterly data is %s\n'
              'The end date of quarterly data is %s\n'
              'The start date of yearly data is %s\n'
              'The end date of yearly data is %s'%(q_start,q_end,a_start,a_end))





        year_lower_space = compute_year(start_year, a_start)
        year_upper_space = compute_year(start_year, a_end)
        quarter_lower_space=compute_quarter(start_quarter,q_start)
        quarter_upper_space=compute_quarter(start_quarter,q_end)



        print('The sector contains: %s' % (option))

        sector = input('Select the sector to visualize [enter esc to quit]:')

        if sector == 'esc':
            break

        if sector in option:

            existing_num = np.load('./Data/CleanData/Existing_num.npy', allow_pickle=True).item()[sector]

            quarter_company_num = existing_num[0][::-1][quarter_lower_space:quarter_upper_space + 1, :][::-1, :]
            annual_company_num = existing_num[1][::-1][year_lower_space:year_upper_space+1, :][::-1, :]



            pickle_in = open(handler.cleanPath + '/cleanData.pickle', 'rb')
            new_dict = pickle.load(pickle_in)

            quarter_count, annual_count,company_num = handler.numpy(new_dict[sector],sector=sector, count_only=True)


            normalized_annual_map=(annual_count/annual_company_num)
            normalized_quarter_map=(quarter_count/quarter_company_num)

            normalized_annual_map[np.isnan(normalized_annual_map)]=1 # deal with divided by zero
            normalized_quarter_map[np.isnan(normalized_quarter_map)] = 1 # deal with divided by zero

            annual_count=annual_count/company_num
            quarter_count=quarter_count/company_num


            while True:

                report_type = input('Select the report type [annual,quarter,esc]:')

                if report_type == 'annual':

                    handler.viusalizeMissingData(sector=sector, X=annual_count)
                    handler.viusalizeMissingData(sector=sector,X=normalized_annual_map)
                elif report_type == 'quarter':

                    handler.viusalizeMissingData(sector=sector, X=quarter_count)
                    handler.viusalizeMissingData(sector=sector, X=normalized_quarter_map)
                elif report_type == 'esc':
                    break

                else:
                    print('Invalid Input, plz try again')
        else:
            print('Plz enter the sector with shown list')


if __name__ == '__main__':
    main()
    # handler = DataHandler()
    # handler.saveStockData(q_start='2010-1', q_end='2019-3', save=True, a_start='2010', a_end='2019')
