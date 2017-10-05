import numpy as np
import pandas as pd
from sklearn import preprocessing

# load data
#crimes05 = pd.read_csv('input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
#crimes08 = pd.read_csv('input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
crimes12 = pd.read_csv('input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

#crimes12 = pd.concat([crimes05, crimes08, crimes12], axis=0)
#print("Dataset from 05 to 07 is:", crimes05.shape, crimes05.head())
#print("Dataset from 05 to 07 is:", crimes08.shape, crimes08.head())
#print("Dataset from 05 to 07 is:", crimes12.shape, crimes12.head())

#discover the features in the data
print ("data before find duplication:", crimes12.shape)
crimes12.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print ("data after find duplication:", crimes12.shape)
print(crimes12.head())

#transform date to datetime Type
crimes12.Date = pd.to_datetime(crimes12.Date, format='%m/%d/%Y %I:%M:%S %p')

#get useful columns
features = pd.DataFrame(columns = ["Date", "Primary Type", "Location Description", "Community Area", "Beat", "District", "Ward"])
#fea_value = crimes12.iloc[:, 4]
#fea_value = crimes12['Date']
fea_value = crimes12[["Date", "Primary Type", "Location Description", "Community Area", "Beat", "District", "Ward"]]
print ("head of feature:", fea_value.head(3))
print ("tail of feature:", fea_value.tail(3))
#fea_value = crimes12.drop(["Unnamed: 0", "ID", "Case Number", "Block", "IUCR", "Description", "Arrest", "Domestic", "Beat", "District", "Ward"], axis = 1)
#features.describe()
print ("selected features are:", fea_value.info())
print ("the number of crime from 12_17 is:", fea_value.shape)

#drop null rows
fea_value2 = fea_value.dropna(axis=0)
#set date as index
fea_value2.index = pd.DatetimeIndex(fea_value2.Date)
print ("selected features 2 are:", fea_value2.info())
print ("possible places are:", fea_value2['Location Description'].value_counts())
print ("primary types are:", fea_value2['Primary Type'].value_counts())

#preprocessing the data for training
#set labels for primary_type
primary_type = preprocessing.LabelEncoder()
fea_value2.loc[:, 'Primary Type in number'] = primary_type.fit_transform(fea_value2['Primary Type'])
print ("labels for primary types are:", fea_value2['Primary Type in number'].value_counts())
#one-hot encoding for primary_type
fea_onecode = fea_value2['Primary Type'].astype(str).str.get_dummies()
#set labels for time in a day
fea_value2['hour'] = fea_value2.index.hour
fea_value2['min'] = fea_value2.index.minute
fea_value2["Business Hour"] =np.where(fea_value2['hour'].between(9,17), 1, 0)
#set labels for day of the week
fea_value2['Day of Week'] = fea_value2.index.dayofweek
fea_value2["Business Day"] =np.where(fea_value2['Day of Week'].between(0,4), 1, 0)
#set day of a month
fea_value2['Day of Month']= fea_value2.index.day

print ("hour and min in a day is:", fea_value2.head())



#set labels for crime spaces
location = fea_value2['Location Description']
print(location.value_counts())
fea_value2.loc[:, 'Location Description Number'] =np.where(location.str.contains('RESIDEN')
                                                        |location.str.contains('APARTMENT'), 1,
                                                     np.where(location.str.contains('STREET')
                                                              |location.str.contains('ALLEY')
                                                              |location.str.contains('SIDEWALK')
                                                              |location.str.contains('LOT')
                                                              |location.str.contains('PARK')
                                                              |location.str.contains('STATION')
                                                              |location.str.contains('PUBLIC')
                                                              |location.str.contains('PLATFORM'), 2,
                                                              np.where(location.str.contains('STORE')
                                                                       |location.str.contains('RESTAURANT')
                                                                       |location.str.contains('SCHOOL')
                                                                       |location.str.contains('BUILDING')
                                                                       |location.str.contains('BAR')
                                                                       |location.str.contains('OFFICE')
                                                                       |location.str.contains('BUS')
                                                                       |location.str.contains('BANK')
                                                                       |location.str.contains('HOTEL')
                                                                       |location.str.contains('TRAIN')
                                                                       |location.str.contains('VEHICLE'), 3, 0)))
##separate into two categories
# fea_value2.loc[:, 'Location Description Number'] =np.where(location.str.contains('RESIDEN')
#                                                         |location.str.contains('APARTMENT'), 1,
#                                                      np.where(location.str.contains('STREET')
#                                                               |location.str.contains('ALLEY')
#                                                               |location.str.contains('SIDEWALK')
#                                                               |location.str.contains('LOT')
#                                                               |location.str.contains('PARK')
#                                                               |location.str.contains('STATION')
#                                                               |location.str.contains('PUBLIC')
#                                                               |location.str.contains('PLATFORM')
#                                                               |location.str.contains('STORE')
#                                                               |location.str.contains('RESTAURANT')
#                                                               |location.str.contains('SCHOOL')
#                                                               |location.str.contains('BUILDING')
#                                                               |location.str.contains('BAR')
#                                                               |location.str.contains('OFFICE')
#                                                               |location.str.contains('BUS')
#                                                               |location.str.contains('BANK')
#                                                               |location.str.contains('HOTEL')
#                                                               |location.str.contains('TRAIN')
#                                                               |location.str.contains('VEHICLE'), 2, 0))


print(fea_value2['Location Description Number'].value_counts(),fea_value2['Location Description'].head(20),
      fea_value2['Location Description Number'].head(20))

selected_features = fea_value2[["Date", "Primary Type in number", "Location Description Number",
                                "Community Area", "hour", "Day of Week", "Business Hour", "Business Day",
                                "Day of Month", "Beat", "District", "Ward"]]
selected_features = pd.concat([fea_onecode, selected_features], axis=1)
print("before drop 0:", selected_features["Location Description Number"].value_counts(), selected_features.info())
selected_features = selected_features[selected_features["Location Description Number"] != 0]
print("After drop 0:", selected_features["Location Description Number"].value_counts(), selected_features.info())

# put features and output labels into a csv file
selected_features.to_csv('selected_features_2012_to_2017.csv')

#seperate according to year
crimes_2014 = fea_value2.loc['2016']
print(crimes_2014.info(), crimes_2014.head())