import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

df = pd.read_csv('Database_No_show_appointments/noshowappointments-kagglev2-may-2016.csv')
df.head()

df.shape

df.info()

df = df.drop(df[df.duplicated()].index)
df.duplicated().sum()

df.describe()

df.rename(columns=lambda x: x.strip().lower(), inplace=True)


df.rename({"no-show": "no_show"}, axis='columns', inplace =True)
df.rename({"hipertension": "hypertension"}, axis='columns', inplace =True)
df.rename({"handcap": "handicap"}, axis='columns', inplace =True)

df['appointmentday'] = pd.to_datetime(df['appointmentday'])
df['month'] = df['appointmentday'].dt.month
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df["diff_between_days"]= (df['appointmentday'].dt.date - df['scheduledday'].dt.date).dt.days

df = df.drop(df[df.age < 1].index)
df.drop(['patientid', 'appointmentid'], axis = 1, inplace = True)

df.head()

df.hist(figsize=(13,13));


def plotting(column_name):
    attended = df.no_show == "No"
    not_attended = df.no_show == "Yes"
    df[column_name][attended].hist(alpha=0.5, bins=10, label='attended',figsize=(13,6))
    df[column_name][not_attended].hist(alpha=0.5, bins=10, label='not_attended',figsize=(13,6))
    plt.legend();
    plt.ylabel('No. of Patients')
    plt.xlabel(column_name + " Condition")
    plt.title('No. of patients relvant to '+ column_name)
    plt.xticks(rotation='vertical')
    
    
plotting("age")

plotting("sms_received")

hypertensionshow =df.groupby(['no_show','hypertension'])['hypertension'].count()["No"]
hypertensionnotshow =df.groupby(['no_show','hypertension'])['hypertension'].count()["Yes"]

ind = np.arange(len(hypertensionshow))
width = 0.35    

red_bars = plt.bar(ind, hypertensionshow, width, color='r', alpha=.7, label='not_hypertension')
black_bars = plt.bar(ind + width, hypertensionnotshow, width, color='black', alpha=.7, label='hypertension')

plt.ylabel('No. of Patients')
plt.xlabel('sms Condition')
plt.title('No. of patients relvant to sms')
locations = ind + width / 2  
labels = ["show","notshow"]  
plt.xticks(locations, labels)
plt.legend()

plotting("diff_between_days")

df00 = df.loc[df.diff_between_days < 0]
df00

from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])