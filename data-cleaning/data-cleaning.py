"""
original post:
https://medium.com/bitgrit-data-science-publication/data-cleaning-with-python-f6bc3da64e45

Definition:
the process of detecting and correcting (or removing) corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('fivethirtyeight')
pd.options.display.width = 0

airbnb_url = 'https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv'
dataset = pd.read_csv(airbnb_url)
rbnb = dataset.copy()

#EDA data
rbnb.head()
rbnb.info()

#I understand that we have some columns with names, host id, some categories, rank values, geoposition...

#separate categorical and numerical columns
cat_dataset = rbnb.select_dtypes(include='object')
num_dataset = rbnb.select_dtypes(exclude='object')

print('Numerical columns:',num_dataset.columns.tolist())
print('Categorical columns:',cat_dataset.columns.tolist())

#MISSING DATA
#I want to visualize and how 'nan' is my data

g1 = sns.heatmap(rbnb.isna(),cmap='viridis',cbar=False)  #visualize nan
g1.set(yticklabels=[])
plt.tight_layout()
plt.show()
rbnb.isna().sum()/rbnb.shape[0]  #percentage of nan
(rbnb.isna().sum()/rbnb.shape[0])[rbnb.isna().sum()/rbnb.shape[0]>0]*100

def missingvalues(tab):
    perc = rbnb.isna().sum()
    return perc[perc>0]

# columns
# name                  0.032723%
# host_name             0.042949%
# last_review          20.558339%
# reviews_per_month    20.558339%

# Ways to deal with missing values:
# 1. Drop the feature (column)
# 2. Drop the row
# 3. Impute the missing value - put a value based on some strategy
# 4. Replace it - put 'VALUE' desired

# It's important to understand that dropping rows also drop important values from other columns.
# My personal strategy would be :
#   drop specific columns with personal info, like host_name. But I don't have columns with this kind.
#   drop columns that are no use or can't be replaced: last_review
#   Don't want to drop any row because some important data can be dropped
#   Replace Name and Host_name to 'None', review_per_month to 0




#try to find the host_name by host_id
rbnb['host_id'][rbnb['host_name'].isna()].value_counts()  #there are 2 ids appears +1
hostIdFromMissingNames = rbnb['host_id'][rbnb['host_name'].isna()].value.tolist()
rbnb['host_name'][rbnb['host_id'].isin(hostIdFromMissingNames)]  #these ids don't have 'host_name'

missingvalues(rbnb)
# so:
# name                  0.032723% - None
# host_name             0.042949% - None
# last_review          20.558339% - drop
# reviews_per_month    20.558339% - 0

rbnb.fillna({'name':'None','host_name':'None','reviews_per_month':0},inplace=True)
rbnb.drop(['last_review'],axis=1,inplace=True)
missingvalues(rbnb)  #empty

#RENAMING COLS
newCols = {'latitude':'lat', 'longitude':'long', 'number_of_reviews':'reviews','calculated_host_listings_count':'host_list_count','availability_365':'availability'}
rbnb.rename(columns=newCols,inplace=True)

#CHANGE DATETYPE
rbnb.dtypes
# id                       int64
# name                    object
# host_id                  int64
# host_name               object
# neighbourhood_group     object
# neighbourhood           object
# lat                    float64
# long                   float64
# room_type               object
# price                    int64
# minimum_nights           int64
# reviews                  int64
# reviews_per_month      float64
# host_list_count          int64
# availability             int64

rbnb.duplicated().any()  #False - no duplicated data

#OUTLIAR
rbnb.describe()  #strange prices and minimum_nights
plt.hist(rbnb['price'][rbnb['price']>200].values,bins=30)  #It's not strange to have high prices per day
sns.jointplot(rbnb['minimum_nights'],rbnb['price'])  #relation btw min_nights and price of the accommodation
plt.hist(rbnb['minimum_nights'][rbnb['minimum_nights']<=30].values,bins=30)  #Clearly I have 2 kind of users: for short period and long period accommodation



