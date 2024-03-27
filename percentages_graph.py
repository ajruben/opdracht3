import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("37296ned_UntypedDataSet_26032024_091942.csv", header = 0, sep=';')
df['Perioden'] = pd.to_datetime(df['Perioden'].str[:4], format='%Y')

#setting divorced, unmarried, married and widowed people in the Netherlands as a percentage of the total population.
df["percentage_divorced"] = df["TotaleBevolking_4"] / df["Gescheiden_8"] * 100
df["percentage_divorced"] = df["percentage_divorced"].round(2)
df["percentage_unmarried"] = df["TotaleBevolking_4"] / df["Ongehuwd_5"] * 100
df["percentage_unmarried"] = df["percentage_unmarried"].round(2)
df["percentage_married"] = df["TotaleBevolking_4"] / df["Gehuwd_6"] * 100
df["percentage_married"] = df["percentage_married"].round(2)
df["percentage_widowed"] = df["TotaleBevolking_4"] / df["Verweduwd_7"] * 100
df["percentage_widowed"] = df["percentage_widowed"].round(2)

#plot the percentage of divorced people in the Netherlands over time. 
df.set_index('Perioden', inplace=True)
plt.plot(df.index, df["percentage_divorced", "percentage_unmarried", "percentage_married", "percentage_widowed"], marker='o', linestyle='-')
plt.title('percentage divorced Over Time')
plt.xlabel('Year')
plt.ylabel("percentage of dutch population")
plt.legend("percentage divorced", "percentage unmarried", "percentage_married", "percentage_widowed")
plt.grid(True)
plt.show()
