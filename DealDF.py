import pandas as pd

def dealDF(symbol_data):



    def findQuarter(date):
        if date.month >= 1 and date.month <= 3:
            return str(date.year) + '-1'
        if date.month >= 4 and date.month <= 6:
            return str(date.year) + '-2'
        if date.month >= 7 and date.month <= 9:
            return str(date.year) + '-3'
        if date.month >= 10 and date.month <= 12:
            return str(date.year) + '-4'

    def findYear(date):
        return str(date.year)



    # Transpose each financial statement
    quarter_earning = pd.DataFrame(symbol_data['Earnings']['History']).T
    quarter_income = pd.DataFrame(symbol_data['Financials']['Income_Statement']['quarterly']).T
    quarter_balance = pd.DataFrame(symbol_data['Financials']['Balance_Sheet']['quarterly']).T
    quarter_cash = pd.DataFrame(symbol_data['Financials']['Cash_Flow']['quarterly']).T
    annual_earning = pd.DataFrame(symbol_data['Earnings']['Annual']).T
    annual_income = pd.DataFrame(symbol_data['Financials']['Income_Statement']['yearly']).T
    annual_balance = pd.DataFrame(symbol_data['Financials']['Balance_Sheet']['yearly']).T
    annual_cash = pd.DataFrame(symbol_data['Financials']['Cash_Flow']['yearly']).T

    # convert the date type
    quarter_earning['date'] = pd.to_datetime(quarter_earning['date'])
    quarter_income['date'] = pd.to_datetime(quarter_income['date'])
    quarter_balance['date'] = pd.to_datetime(quarter_balance['date'])
    quarter_cash['date'] = pd.to_datetime(quarter_cash['date'])

    # change the yearly date to standard year
    annual_earning['date'] = pd.to_datetime(annual_earning['date'])
    annual_income['date'] = pd.to_datetime(annual_income['date'])
    annual_balance['date'] = pd.to_datetime(annual_balance['date'])
    annual_cash['date'] = pd.to_datetime(annual_cash['date'])

    # select the quartely report <2022.1
    quarter_earning = quarter_earning[(quarter_earning['date'].dt.year <= 2021) | (
                (quarter_earning['date'].dt.year == 2022) & (quarter_earning['date'].dt.month <= 3))]
    annual_income = annual_income[(annual_income['date'].dt.year <= 2021) | (
                (annual_income['date'].dt.year == 2022) & (annual_income['date'].dt.month <= 3))]
    annual_balance = annual_balance[(annual_balance['date'].dt.year <= 2021) | (
                (annual_balance['date'].dt.year == 2022) & (annual_balance['date'].dt.month <= 3))]
    annual_cash = annual_cash[(annual_cash['date'].dt.year <= 2021) | (
                (annual_cash['date'].dt.year == 2022) & (annual_cash['date'].dt.month <= 3))]

    annual_earning = annual_earning[annual_earning['date'].dt.year <= 2021]
    annual_income = annual_income[annual_income['date'].dt.year <= 2021]
    annual_balance = annual_balance[annual_balance['date'].dt.year <= 2021]
    annual_cash = annual_cash[annual_cash['date'].dt.year <= 2021]

    # change the quarterly date to standard quarter
    quarter_earning['date'] = (quarter_earning['date']).apply(findQuarter)
    quarter_income['date'] = (quarter_income['date']).apply(findQuarter)
    quarter_balance['date'] = (quarter_balance['date']).apply(findQuarter)
    quarter_cash['date'] = (quarter_cash['date']).apply(findQuarter)

    # change the yearly date to standard year
    annual_earning['date'] = (annual_earning['date']).apply(findYear)
    annual_income['date'] = (annual_income['date']).apply(findYear)
    annual_balance['date'] = (annual_balance['date']).apply(findYear)
    annual_cash['date'] = (annual_cash['date']).apply(findYear)

    # Merge several quarterly reports into one
    quarter_data = pd.merge(quarter_earning, quarter_income, on='date', how='outer')
    quarter_data = pd.merge(quarter_data, quarter_balance, on='date', how='outer')
    quarter_data = pd.merge(quarter_data, quarter_cash, on='date', how='outer')
    quarter_data = pd.merge(quarter_data, quarter_earning, on='date', how='outer')

    # Merge several yearly reports into one
    annal_data = pd.merge(annual_earning, annual_income, on='date', how='outer')
    annal_data = pd.merge(annal_data, annual_balance, on='date', how='outer')
    annal_data = pd.merge(annal_data, annual_cash, on='date', how='outer')
    annal_data = pd.merge(annal_data, annual_earning, on='date', how='outer')  # Ctrl+w 选中var

    return quarter_data,annal_data