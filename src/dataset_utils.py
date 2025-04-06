import pandas as pd

class Dataset():

    def __init__(self, years_to_target):

        """
        Instantiate a Dataset object which has the following attributes:
        - years: prediction horizon, used to build the target variable for a given observation
        - merged_data: the dataset build using the data sources, but w/o target
        - data: final dataset with the target variable
        """
        
        self.years = years_to_target
        self.merged_data = None
        self.data = None



    def combine_data_sources(self):

        """
        Combines the different sources of data into one single dataset
        """

        print('- combining the different data sources..')

        # read the data sources and create pandas dataframes
        balance_sheet_merged_data = pd.read_csv('../data/raw/balance_sheet_annual.csv')
        income_statement_merged_data = pd.read_csv('../data/raw/income_statement_annual.csv')
        eps_data = pd.read_csv('../data/raw/earnings_annual.csv')
        overview_data = pd.read_csv('../data/raw/company_overview.csv')
        
        # drop the currency columns which will not be of any use
        balance_sheet = balance_sheet_merged_data.drop('reportedCurrency', axis=1)
        income_statement = income_statement_merged_data.drop('reportedCurrency', axis=1)

        # merging balance sheet and income statement merged_data
        print('---- merging balance sheet and income statement data..')
        self.merged_data = balance_sheet.merge(income_statement, how='inner', on=['symbol', 'fiscalDateEnding'])

        # adding EPS merged_data
        # will require several steps as the dates from the different files do necessarily match perfectly
        # first try merging using exact date
        print('---- adding the EPS data..')
        print('------ Using the exact date..')
        self.merged_data = self.merged_data.merge(eps_data, on=['fiscalDateEnding', 'symbol'], how='left')
        print(f"-------- missing eps values: {round(100 * self.merged_data['reportedEPS'].isna().sum() / len(self.merged_data['reportedEPS'].values),2)}%")


        
        # then try merging only the year and month
        print('------ using the year and the month only..')
        self.merged_data['YM_date'] = self.merged_data['fiscalDateEnding'].str[:7]
        eps_temp = eps_data.copy()
        eps_temp['YM_date'] = eps_temp['fiscalDateEnding'].str[:7]
        eps_temp.drop('fiscalDateEnding', axis=1, inplace=True)
        self.merged_data = self.merged_data.merge(eps_temp, on=['symbol', 'YM_date'], how='left').drop('YM_date', axis=1)
        self.merged_data.loc[self.merged_data['reportedEPS_x'].isna(), 'reportedEPS_x'] = self.merged_data.loc[self.merged_data['reportedEPS_x'].isna(), 'reportedEPS_y']
        self.merged_data.drop('reportedEPS_y', axis=1, inplace=True)
        print(f"-------- % missing eps values: {round(100 * self.merged_data['reportedEPS_x'].isna().sum() / len(self.merged_data['reportedEPS_x'].values),2)}%")

        # finally, we can try merging using the year only
        print('------ using the year only..')
        self.merged_data['Y_date'] = self.merged_data['fiscalDateEnding'].str[:4]
        eps_temp = eps_data.copy()
        eps_temp['Y_date'] = eps_temp['fiscalDateEnding'].str[:4]
        eps_temp.drop('fiscalDateEnding', axis=1, inplace=True)
        eps_temp = eps_temp.groupby(['Y_date', 'symbol'], as_index=False).max()
        self.merged_data = self.merged_data.merge(eps_temp, on=['symbol', 'Y_date'], how='inner').drop('Y_date', axis=1)
        self.merged_data.loc[self.merged_data['reportedEPS_x'].isna(), 'reportedEPS_x'] = self.merged_data.loc[self.merged_data['reportedEPS_x'].isna(), 'reportedEPS']
        self.merged_data.drop('reportedEPS', axis=1, inplace=True)
        self.merged_data.rename(columns={'reportedEPS_x': 'reportedEPS'}, inplace=True)
        print(f"-------- missing eps values: {round(100 * self.merged_data['reportedEPS'].isna().sum() / len(self.merged_data['reportedEPS'].values),2)}%")

        # add sector and industry 
        print('---- adding the sector and industry information..')
        overview = overview_data[['symbol', 'Sector', 'Industry']]
        self.merged_data = self.merged_data.merge(overview, on='symbol', how='left')

        # save merged_data
        self.merged_data.to_csv('../data/preprocessed/preprocessed_data.csv', index=False)
        print('---- done')



    def create_target_variable(self):

        """
        Creates the target variable using the "year" attribute from the dataset object
        """

        print('- creating the target variable')

        # extracting the year from the merged_data        
        # testing whether there is no duplicate year for each ticker
        self.data = self.merged_data.copy()
        self.data['year'] = self.data.fiscalDateEnding.str[:4].astype(int)
        temp = self.data.copy()
        temp = temp[['symbol', 'fiscalDateEnding', 'year']]
        symbols = temp['symbol'].unique()
        aggregation_required = False
        print('---- checking for duplicate ticker/symbol combinations..')
        for sym in symbols:
            listofyears = temp[temp['symbol'] == sym]['year'].values
            setofyears = set(listofyears)
            if len(listofyears) != len(setofyears):
                aggregation_required = True
                print(f"------ symbol: {sym} | list length: {len(listofyears)} | set length: {len(setofyears)} | duplicate(s): {len(listofyears) - len(setofyears)}")
        # aggregating the merged_data in case there is a duplicate
        if aggregation_required:
            print('------ aggregating the duplicates..')
            self.data = self.data.groupby(['year', 'symbol'], as_index=False).max()
        else:
            print('------ no duplicate found')
        
        target_merged_data = self.data.copy()
        target_merged_data = target_merged_data[['symbol', 'year', 'reportedEPS']]
        # we can substract the prediction horizon from the year
        target_merged_data['year'] = target_merged_data['year'] - self.years
        # let's also rename the columns
        target_merged_data.rename(columns={'reportedEPS' : 'futureEPS'}, inplace=True)
        # Now we can merge the original merged_dataset with this new one
        self.data = self.data.merge(target_merged_data, on=['symbol', 'year'], how='left')

        # saving the merged_data
        self.data.to_csv('../data/preprocessed/preprocessed_data_with_target.csv', index=False)
        print('---- done')

    

    def create_dataset(self):

        """
        Combines the previous two operations
        """

        self.combine_data_sources()
        self.create_target_variable()
