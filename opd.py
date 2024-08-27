import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
# data = pd.read_csv('september.csv')

# #print
# print(data.head())


df = pd.read_csv('september_192223.csv')

print(df.head())

#print colom
print(df.columns)


# Add new columns for 2024
df['OPD Seen - Sept - 2024_New'] = None  
df['OPD Seen - Sept - 2024_Review'] = None  
df['OPD Seen - Sept - 2024_Total'] = None  

# Calculate averages for the 2019  2022 2023
df['Average_OPD Seen_New'] = df[['OPD Seen-Sept-2019_New', 'OPD Seen-Sept-2022_New', 'OPD Seen-Sept-2023_New']].mean(axis=1)
df['Average_OPD Seen_Review'] = df[['OPD Seen-Sept-2019_Review', 'OPD Seen-Sept-2022_Review', 'OPD Seen-Sept-2023_Review']].mean(axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('september_updated.csv', index=False)

#display complete data
print(df.head())
#print last 3 columns
print(df[['Average_OPD Seen_New']].head())


# Calculate the average growth from 2022 to 2023 that is OPD Seen-Sept-2022_New and OPD Seen-Sept-2023_New
df['Average_Growth_New_2219'] = (df['OPD Seen-Sept-2022_New'] - df['OPD Seen-Sept-2019_New']) / df['OPD Seen-Sept-2019_New'] * 100
df['Average_Growth_New_2322'] = (df['OPD Seen-Sept-2023_New'] - df['OPD Seen-Sept-2022_New']) / df['OPD Seen-Sept-2022_New'] * 100
df['Average_Growth_New_2319'] = (df['OPD Seen-Sept-2023_New'] - df['OPD Seen-Sept-2019_New']) / df['OPD Seen-Sept-2019_New'] * 100

# Calculate the average growth from 2022 to 2023 that is OPD Seen-Sept-2022_Review and OPD Seen-Sept-2023_Review
df['Average_Growth_Review_2219'] = (df['OPD Seen-Sept-2022_Review'] - df['OPD Seen-Sept-2019_Review']) / df['OPD Seen-Sept-2019_Review'] * 100
df['Average_Growth_Review_2322'] = (df['OPD Seen-Sept-2023_Review'] - df['OPD Seen-Sept-2022_Review']) / df['OPD Seen-Sept-2022_Review'] * 100
df['Average_Growth_Review_2319'] = (df['OPD Seen-Sept-2023_Review'] - df['OPD Seen-Sept-2019_Review']) / df['OPD Seen-Sept-2019_Review'] * 100

#average growth rate for new and review
df['Average_Growth_New'] = (df['Average_Growth_New_2219'] + df['Average_Growth_New_2322'] + df['Average_Growth_New_2319'])/3
df['Average_Growth_Review'] = (df['Average_Growth_Review_2219'] + df['Average_Growth_Review_2322'] + df['Average_Growth_Review_2319'])/3

#calculate the 2024 average OPD Seen_total using avg of new and review and average growth
df['Expected_OPD_2024_New'] =((df['Average_OPD Seen_New'] *(df['Average_Growth_New']))/100) +df['Average_OPD Seen_New']


#review
df['Expected_OPD_2024_Review'] =((df['Average_OPD Seen_Review'] *(df['Average_Growth_Review']))/100) +df['Average_OPD Seen_Review']

#calculate the total
df['Expected_OPD_2024_Total'] = df['Expected_OPD_2024_New'] + df['Expected_OPD_2024_Review']

# Save the updated DataFrame to a new CSV file
df.to_csv('september_updated.csv', index=False)









# Save the updated DataFrame to a new CSV file
df.to_csv('september_updated.csv', index=False)

# Display the updated DataFrame
print(df.head())









