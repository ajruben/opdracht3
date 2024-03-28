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
        df_log[column + '_log'] = (np.log(df[column]/df[column].shift(1))).dropna()
        nreturn = ((df[column] - df[column].shift(1))/df[column].shift(1)).dropna()
        df_nreturn[column + '_nreturn'] = nreturn
        n2return = (nreturn - nreturn.shift(1)/nreturn.shift(1)).dropna()
        df_n2return[column + '_n2return'] = n2return
    return df_log, df_nreturn, df_n2return


df2 = df.select_dtypes(exclude=['object', 'string'])
df3 = df.select_dtypes(include=['object', 'string'])

def convert_to_float(col):
    return pd.to_numeric(col, errors='coerce')

# Apply th
# e function to the entire DataFrame
df3 = df3.map(convert_to_float)
df3.astype(float)


df = pd.concat([df2, df3], axis=1)
df_log, df_nreturn, df_n2return = get_return(df)
print(df_log, df_nreturn, df_n2return)



if not os.path.exists('plots'):
    os.makedirs('plots')

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
