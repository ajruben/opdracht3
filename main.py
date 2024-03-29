import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("37296ned_UntypedDataSet_26032024_091942.csv", header = 0, sep=';')
df['Perioden'] = pd.to_datetime(df['Perioden'].str[:4], format='%Y')

print(df.loc[8])


df.set_index('Perioden', inplace=True)
#dit plot alle columns, je kunt ook een list maken met de namen van de columns, en dan loopen over die list en dan column = df[loop]

def get_return(df):
    columns = list(df.columns)
    
    df_log = pd.DataFrame()
    df_nreturn = pd.DataFrame()
    df_n2return = pd.DataFrame()

    for column in columns:
        #get returns
        logreturn = (np.log(df[column]/df[column].shift(1))).dropna()        
        nreturn = ((df[column] - df[column].shift(1))/df[column].shift(1)).dropna()
        n2return = (nreturn - nreturn.shift(1)/nreturn.shift(1)).dropna()
        #put into dfs
        df_log[column + '_log'] = logreturn
        df_nreturn[column + '_nreturn'] = nreturn
        df_n2return[column + '_n2return'] = n2return
        big_value = 1
        df_log.replace([np.inf, -np.inf], big_value, inplace=True)
        df_nreturn.replace([np.inf, -np.inf], big_value, inplace=True)
        df_n2return.replace([np.inf, -np.inf], big_value, inplace=True)
    return df_log, df_nreturn, df_n2return, logreturn, nreturn, n2return



df2 = df.select_dtypes(exclude=['object', 'string'])
df3 = df.select_dtypes(include=['object', 'string'])

def convert_to_float(col):
    return pd.to_numeric(col, errors='coerce')

# Apply th
# e function to the entire DataFrame
df3 = df3.map(convert_to_float)
df3.astype(float)


df = pd.concat([df2, df3], axis=1)
df_log, df_nreturn, df_n2return, logreturn, nreturn, n2return = get_return(df)

if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots_returns'):
     os.makedirs('plots_returns')

for column in df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)
    plt.title(f'{column} Over Time')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{column}_over_time.png')  # Save the plot in the 'plots' directory
    plt.close() 



def linear_regression(df,column1, column2): 
        """
        background: https://www.probabilitycourse.com/chapter8/8_5_2_first_method_for_finding_beta.php
        """
        lin_dict = {}
        column1_name=column1
        column2_name=column2
        column1 = df.loc[:, column1].values
        column2 = df.loc[:, column2].values
        #all calculations
        cov = np.cov(column1, column2)
        beta1 = cov[1,0]/cov[0,0]
        beta0 =  column1.mean() - beta1*column2.mean()
        r_sq = (cov[1,0])**2 /(cov[0,0]*cov[1,1])
        vol_daily = column2.std()
        vol_period = vol_daily*np.sqrt(len(column2))
        #store results
        linear_regression_ = {'covmat':cov, 'beta1' : beta1, 'beta0' : beta0, 'r_sq' : r_sq, 'vol_daily' : vol_daily, 'vol_period': vol_period, 
                                'column1':column1, 'column2':column2}#, 'company_returns':company, 'index_returns':index}
    
        #init figure, create scatter
        plt.figure()
        plt.scatter(column2, column1, label='return on a given year', color='blue', marker='.')

        #borders and regression line
        x_min = (column2.min()) - (column2.max()*0.2)
        x_max = (column2.max()) * 1.2
        y_min = (column1.min()) - (column1.max()*0.2)
        y_max = (column1.max()) * 1.2
        x_line = np.linspace(x_min-1, x_max+1, num=2)
        y_line = beta0 + (beta1 * x_line)  
        plt.plot(x_line, y_line, label='Linear regression', color='red')
        
        # layout
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(f'{column2_name}-axis')
        plt.ylabel(f'{column1_name}-axis')
        plt.title(f'Linear regression for: {column1_name} and {column2_name}')
        plt.grid()
        plt.legend()

        # Save and close
        plt.savefig(f'plots_returns/linear_regression_{column1_name}_{column2_name}.png',
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

n_columns=list(df_nreturn.columns)
df_columns = list(df.columns)

for outer_column in range(len(n_columns) - 1):
    if not df[df_columns[outer_column]].isnull().any().any():
        for inner_column in range(outer_column, len(n_columns)):
            if not df[df_columns[inner_column]].isnull().any().any():
                linear_regression(df_nreturn, n_columns[outer_column], n_columns[inner_column])
                print(f"done {outer_column} and {inner_column}")