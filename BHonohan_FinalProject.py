'''
Created on July 19, 2021
@author: Blaine Honohan
CS602 Summer 2021 - Final Project - Boston Crime Data
'''

#import packages needed
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.models.tools import HoverTool

#getData function creates dataframes that will be used in the rest of the site/analysis
def getData():
    #crime data file
    crime_file = pd.read_csv('bostoncrime2021_7000_sample.csv')
    crime_df = pd.DataFrame(crime_file)

    #districts file that will be used for merge
    districts_file = pd.read_csv('BostonPoliceDistricts.csv')
    districts_df = pd.DataFrame(districts_file)

    #merge two files on Distric to bring in the District Name
    boston_crime = pd.merge(crime_df, districts_df, how='left', left_on='DISTRICT', right_on='District')

    #drop district code from districts_df (duplicate column)
    boston_crime.drop(['District'], axis=1, inplace=True)

    #sort by District Name and Date
    boston_crime.sort_values(by=['District Name','OCCURRED_ON_DATE'], inplace=True)

    #fill NaN with N/A
    boston_crime['District Name'].fillna('N/A', inplace=True)

    #replace latitude and longitude names for map
    boston_crime.rename(columns={'Lat': 'latitude', 'Long': 'longitude'}, inplace=True)

    #create time of day column to use in app
    time_conditions = [
        (boston_crime['HOUR'] >= 5) & (boston_crime['HOUR'] < 12),
        (boston_crime['HOUR'] >= 12) & (boston_crime['HOUR'] < 17),
         (boston_crime['HOUR'] >= 17) & (boston_crime['HOUR'] < 21)]
    times = ['Morning', 'Afternoon', 'Evening']
    boston_crime['Time_Of_Day'] = np.select(time_conditions, times, default='Night')

    #create datetime column based on OCCURRED_ON_DATE and then create new Date column
    boston_crime['Datetime'] = pd.to_datetime(boston_crime['OCCURRED_ON_DATE'])
    boston_crime['Date'] = boston_crime['Datetime'].dt.strftime('%m/%d')

    #min/max month in file to be used as a filter
    min_month = boston_crime['MONTH'].min()
    max_month = boston_crime['MONTH'].max()

    #covid vaccination phases to later be joined into main dataframe
    covid_file = pd.read_csv('covid_vaccine_phases.csv')
    covid_df = pd.DataFrame(covid_file)
    covid_df['Datetime'] = pd.to_datetime(covid_df['Covid_Phase_Date'])
    covid_df['Date'] = covid_df['Datetime'].dt.strftime('%m/%d')

    return boston_crime, min_month, max_month, covid_df

#take user filters and build dataframe with those filters applied
def filteredData(df, district, offense, shooting, start, end, days, time):
    if district != 'All':
        df = df.loc[df['District Name'] == district]

    if offense != 'All':
        df = df.loc[df['OFFENSE_DESCRIPTION'] == offense]

    if shooting:
        df = df[df['SHOOTING'] == 1]

    df = df[df['MONTH'] >= start]

    df = df[df['MONTH'] <= end]

    df = df[df['DAY_OF_WEEK'].isin(days)]

    df = df[df['Time_Of_Day'].isin(time)]

    return df

#build bar chart that will show number of offenses by district
def barChart(df, x_value = 'District Name'):
    df = df[['INCIDENT_NUMBER',x_value]]
    district_gb = df.groupby(by=x_value).count().reset_index()
    district_df = pd.DataFrame(district_gb)
    district_df.columns = [x_value,'Count']

    district_df.sort_values(by='Count', ascending=False, inplace=True)

    source = ColumnDataSource(district_df)
    x_values = district_df[x_value].tolist()

    if x_value == 'District Name':
        plt_title='District Bar Chart'
        x_axis_lbl=x_value
    else:
        plt_title='Offense Bar Chart'
        x_axis_lbl='Offense Description'

    p = figure(
        title=plt_title,
        x_axis_label=x_axis_lbl,
        y_axis_label='# of Crimes',
        x_range=x_values,
        match_aspect=True,
        toolbar_location=None,
        tools=''
    )

    p.vbar(x=x_value, top='Count', width=.7, source=source)

    p.xaxis.major_label_orientation = 'vertical'
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    hover = HoverTool()
    hover.tooltips = [
        ('Total', '@Count')]

    hover.mode = 'vline'

    p.add_tools(hover)

    return p

#build line chart that will show number of offenses by month/date (depending how filtered)
def lineChart(df, covid_df, x_value = 'MONTH'):
    df = df[['INCIDENT_NUMBER',x_value]]

    if x_value == 'MONTH':
        df['MONTH'] = df['MONTH'].astype(str)

    df = df.groupby(by=x_value).count().reset_index()
    df = pd.DataFrame(df)
    df.columns = ['Date','Count']

    df.sort_values(by='Date', inplace=True)

    x_values = list(df['Date'].unique())

    if x_value == 'MONTH':
        x_label = 'Month'
    else:
        x_label = 'Date'

    p = figure(
        title='Crimes by Date',
        x_axis_label=x_label,
        y_axis_label='# of Crimes',
        x_range=x_values,
        match_aspect=True,
        toolbar_location=None,
        tools=''
    )

    source = ColumnDataSource(df)
    p.line(x='Date', y='Count', line_width=2, source=source)

    hover = HoverTool()
    hover.tooltips = [
        ('Total', '@Count')]

    hover.mode = 'vline'

    p.add_tools(hover)

    #add in covid data if actually looking at one specific month.  Covid phases will be added as a secondary axis and always show a value of 1.
    if x_value != 'MONTH':
        df = pd.merge(df, covid_df, how='left', right_on='Date', left_on='Date')
        df = df.dropna()
        source_covid = ColumnDataSource(df)

        p.xaxis.major_label_orientation = 'vertical'

        p.extra_y_ranges = {'covid': Range1d(start=0, end=1)}
        p.add_layout(LinearAxis(y_range_name='covid'), 'right')
        p.vbar(x=x_value, top=1, width=.2, source=source_covid, y_range_name='covid', color='lightblue')

    return p

#build dictionary of the available districts to use for filtering - pre-set the filer with All
def getDistricts(df):
    districts = {'All':0}

    district_count = 1

    for district in df['District Name']:
        if district not in districts.keys() and district != 'N/A':
            districts[district] = district_count
            district_count += 1

    return list(districts.keys())

#build dictionary of offenses, filtered by whatever district is selected
def getOffenses(df, filter):
    df.sort_values(by=['OFFENSE_DESCRIPTION'], inplace=True)

    if filter != 'All':
        df = df[df['District Name'] == filter]

    offenses = {'All':0}

    offense_count = 1

    for offense in df['OFFENSE_DESCRIPTION']:
        if offense not in offenses.keys():
            offenses[offense] = offense_count
            offense_count += 1

    return list(offenses.keys())

#build map dataframe (latitude and longitude) and remove anything with 0 location
def mapData(df):
    map_df = df[df['latitude'] != 0]

    #median for final map
    med_long = map_df['longitude'].median()
    med_lat = map_df['latitude'].median()

    return map_df, med_long, med_lat

#get all the base dataframes we will need for app
boston_crime, min_month, max_month, covid_df = getData()

#title of dashboard
st.title('2021 Boston Crime Data')

#build sidebar/filters
st.sidebar.header('Filters')

#get names of districts and put in a selectbox
district_names = getDistricts(boston_crime)
district = st.sidebar.selectbox('Select District', district_names)

#get names of offenses and put in a selectbox
offense_names = getOffenses(boston_crime, district)
offense = st.sidebar.selectbox('Select Offense Type', offense_names)

#checkbox if you want to only view incidents with a shooting, defaults unchecked
only_shootings = st.sidebar.checkbox('Shootings Only', False)

#hr_start, hr_end = st.sidebar.slider('Choose time of day crime was committed', 0, 23, (0,23), 1)

#checkbox to allow user to filter one month at a time or select multiple months with a slider
#selection will change line chart
mnth_box = st.sidebar.checkbox('One Month at a Time?', False)

if mnth_box:
    mnth_start = st.sidebar.selectbox('Choose Month', list(range(1, 7, 1)))
    mnth_end = mnth_start
else:
    mnth_start, mnth_end = st.sidebar.slider('Choose Month', min_month, max_month, (min_month, max_month), 1)

#build days of the week filter
days_of_the_week = st.sidebar.multiselect('Days of the Week',['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
                                          ,['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

#build time of day filter
time_of_day = st.sidebar.multiselect('Time of Day',['Morning','Afternoon','Evening','Night']
                                     ,['Morning','Afternoon','Evening','Night'])

#explanation of what the times of day are
st.sidebar.write('Morning - 5AM to Noon')
st.sidebar.write('Afternoon - Noon to 5PM')
st.sidebar.write('Evening - 5PM to 9PM')
st.sidebar.write('Night - 9PM to 5AM')

#filter the data based on selections and spit out new dataframe
boston_crime = filteredData(boston_crime, district, offense, only_shootings, mnth_start, mnth_end, days_of_the_week, time_of_day)

#see if the filters actually bring back any data
if boston_crime.empty:
    st.header('No data found for selected filters.  Please try again.')
else:
    #build map
    st.header('Crime Map')
    st.write('Map of crimes in Boston based on selected filters.  Records with no Latitude/Longitude values are excluded.')

    map_df, med_long, med_lat = mapData(boston_crime)

    st.map(map_df)

    #build bar chart - will either by by district or if only one district selected, by offense
    if district == 'All':
        st.header('Crimes by District')
        st.write('Bar chart of offenses by district.')
        p = barChart(boston_crime)
    else:
        st.header(f'Crimes by Offense in {district}')
        st.write('Bar chart of offenses by district.')
        p = barChart(boston_crime, 'OFFENSE_DESCRIPTION')

    st.bokeh_chart(p)

    #build line chard
    st.header('Crimes by Date Analysis')
    st.write('Line chart displaying number of crimes per date.')

    if mnth_start != mnth_end:
        p = lineChart(boston_crime, covid_df)
    else:
        p = lineChart(boston_crime, covid_df, 'Date')

    st.bokeh_chart(p)
