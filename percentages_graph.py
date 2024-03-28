import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("37296ned_UntypedDataSet_26032024_091942.csv", header = 0, sep=';')
df['Perioden'] = pd.to_datetime(df['Perioden'].str[:4], format='%Y')

#setting divorced, unmarried, married and widowed people in the Netherlands as a percentage of the total population.
df["Percentage_Divorced"] = df["Gescheiden_8"] / df["TotaleBevolking_4"] * 100
df["Percentage_Divorced"] = df["Percentage_Divorced"].round(2)
df["Percentage_NeverMarried"] = df["Ongehuwd_5"] / df["TotaleBevolking_4"] * 100
df["Percentage_NeverMarried"] = df["Percentage_NeverMarried"].round(2)
df["Percentage_Married"] = df["Gehuwd_6"] / df["TotaleBevolking_4"] * 100
df["Percentage_Married"] = df["Percentage_Married"].round(2)
df["Percentage_Widowed"] = df["Verweduwd_7"] / df["TotaleBevolking_4"] * 100
df["Percentage_Widowed"] = df["Percentage_Widowed"].round(2)

print(df)

#plot the percentage of divorced people in the Netherlands over time. 
df.set_index('Perioden', inplace=True)
plt.plot(x=df.index, y=df["Percentage_Divorced", "Percentage_NeverMarried", "Percentage_Married", "Percentage_Widowed"], marker='o', linestyle='-')
plt.title('Percentages Over Time')
plt.xlabel('Year')
plt.ylabel("% of Dutch Population")
plt.legend("Percentage Divorced", "Percentage Unmarried", "Percentage Married", "Percentage Widowed")
plt.grid(True)
plt.show()
