
# coding: utf-8

# In[134]:


import os
import matplotlib.pyplot as plt


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:/Users/ASUS/Desktop/ACF')


# In[4]:


os.getcswd()


# In[5]:


os.getcwd()


# In[6]:


import pandas as pd
import numpy as np


# In[7]:


events=pd.read_csv('athlete_events.csv')
regions=pd.read_csv('noc_regions.csv')


# In[8]:


events.head()


# In[9]:


events.info()


# In[10]:


glimpse(events)


# In[12]:


#Factoring null values


# In[13]:


print(events.isnull().sum())


# In[15]:


events['Medal'].fillna('DNW',inplace=True)


# In[16]:


events.info()


# In[17]:


events.head()


# In[18]:


print(events.loc[:,['NOC','Team']].drop_duplicates()['NOC'].value_counts().head())


# In[20]:


#noc_country mapping
regions.head()
regions.drop('notes',axis=1,inplace=True)
regions.rename(columns={'region':'Country'},inplace=True)
regions.head()


# In[32]:


#merging noc data with Olympics Events data
events_merge=events.merge(regions,
                         left_on='NOC',
                          right_on='NOC',
                         how='left')


# In[33]:


events_merge.head()


# In[34]:


events_merge.shape


# In[35]:


events.shape


# In[36]:


regions.shape


# In[37]:


events_merge['NOC'].nunique()


# In[38]:


events_merge.loc[events_merge['Country'].isnull(),['NOC','Team']].drop_duplicates()


# In[39]:


null_values=events_merge['Country'].isnull()


# In[40]:


null_values


# In[41]:


null_values.info


# In[42]:


null_values.unique()


# In[43]:


null_values.nunique()


# In[44]:


print(null_values==True)


# In[55]:


events_merge.loc[events_merge['Country'].isnull()==True,['NOC','Team']].drop_duplicates()



# In[56]:


events_merge['Country']=np.where(events_merge['NOC']=='SGP','Singapore',events_merge['Country'])
events_merge['Country']=np.where(events_merge['NOC']=='ROT','Refugee Olympics Athlete',events_merge['Country'])
events_merge['Country']=np.where(events_merge['NOC']=='UNK','Unknown',events_merge['Country'])
events_merge['Country']=np.where(events_merge['NOC']=='TUV','Tuvalu',events_merge['Country'])
events_merge.drop('Team',axis=1,inplace=True)
events_merge.rename(columns={'Country':'Team'},inplace=True)


# In[59]:


events_merge.head()
events_merge.info()
print(events_merge.isnull().sum())


# In[60]:


print(events.isnull().sum())


# In[61]:


print(events_merge.loc[:,['NOC','Team']].drop_duplicates()['NOC'].value_counts().head())


# In[64]:


gdp=pd.read_csv('world_gdp.csv',skiprows=3)
gdp.drop(['Indicator Name','Indicator Code'],axis=1,inplace=True)


# In[65]:


gdp.head()


# In[67]:


gdp=pd.melt(gdp,id_vars=['Country Name','Country Code'],var_name='Year',value_name='GDP')


# In[68]:


gdp.head()


# In[72]:


events_merge_ccode = events_merge.merge(gdp[['Country Name', 'Country Code']].drop_duplicates(),
                                            left_on = 'Team',
                                            right_on = 'Country Name',
                                            how = 'left')

events_merge_ccode.drop('Country Name', axis = 1, inplace = True)

# Merge to get gdp too


# In[73]:


events_merge_gdp=events_merge_ccode.merge(gdp,left_on=['Country Code','Year'],right_on=['Country Code','Year'],how='left')


# In[74]:


events_merge_ccode['Year']=events_merge_ccode['Year'].astype(int)
gdp['Year']=gdp['Year'].astype('int')


# In[75]:


events_merge_gdp=events_merge_ccode.merge(gdp,left_on=['Country Code','Year'],right_on=['Country Code','Year'],how='left')


# In[77]:


events_merge_gdp.drop('Country Name',axis=1,inplace=True)


# In[78]:


events_merge_gdp.head(2)


# In[79]:


os.chdir('C:/Users/ASUS/Desktop/ACF/Olympics Data Analysis')


# In[84]:


pop=pd.read_csv('world_pop.csv')
pop.drop(['Indicator Name','Indicator Code'],axis=1,inplace=True)
pop=pop.melt(id_vars=['Country','Country Code'],var_name='Year',value_name='Population')
pop.head(2)


# In[85]:


(events_merge_gdp['Country Code'].nunique())-(pop['Country Code'].nunique())


# In[86]:


events_complete=pd.merge(events_merge_gdp,pop,left_on=['Country Code','Year'],right_on=['Country Code','Year'],how='left')


# In[87]:


pop['Year']=pop['Year'].astype(int)
events_merge_gdp['Year']=events_merge_gdp['Year'].astype('int')


# In[88]:


events_complete=pd.merge(events_merge_gdp,pop,left_on=['Country Code','Year'],right_on=['Country Code','Year'],how='left')


# In[89]:


events_complete.head(
)


# In[91]:


events_complete.drop('Country',axis=1,inplace=True)


# In[92]:


events_complete.isnull().sum()


# In[95]:


#Lets take data from 1961 onwards only and for summer olympics only
events_complete_subset = events_complete.loc[(events_complete['Year'] > 1960) & (events_complete['Season'] == "Summer"), :]

# Reset row indices
events_complete_subset = events_complete_subset.reset_index()


# In[97]:


events_complete_subset['Medals_Won']=np.where(events_complete_subset.loc[:,'Medal']=='DNW',0,1)


# In[110]:


identify_team_events = pd.pivot_table(events_complete_subset,
                                      index = ['Team', 'Year', 'Event'],
                                      columns = 'Medal',
                                      values = 'Medals_Won',
                                      aggfunc = 'sum',
                                     fill_value = 0).drop('DNW', axis = 1).reset_index()


# In[111]:


identify_team_events.head()


# In[101]:


events_complete_subset


# In[114]:


identify_team_events = identify_team_events.loc[identify_team_events['Gold'] > 1, :]
team_sports = identify_team_events['Event'].unique()


# In[116]:


identify_team_events



# In[107]:


team_sports


# In[108]:


identify_team_events.nunique()


# In[120]:


identify_team_events.loc[identify_team_events['Event']=="Swimming Men's 50 metres Freestyle"]


# In[121]:


remove_sports = ["Gymnastics Women's Balance Beam", "Gymnastics Men's Horizontal Bar", 
                 "Swimming Women's 100 metres Freestyle", "Swimming Men's 50 metres Freestyle"]

team_sports = list(set(team_sports) - set(remove_sports))


# In[122]:


team_sports


# In[125]:


team_event_mask = events_complete_subset['Event'].map(lambda x: x in team_sports)
single_event_mask = [not i for i in team_event_mask]

# rows where medal_won is 1
medal_mask = events_complete_subset['Medals_Won'] == 1

# Put 1 under team event if medal is won and event in team event list
events_complete_subset['Team_Event'] = np.where(team_event_mask & medal_mask, 1, 0)

# Put 1 under singles event if medal is won and event not in team event list
events_complete_subset['Single_Event'] = np.where(single_event_mask & medal_mask, 1, 0)

# Add an identifier for team/single event
events_complete_subset['Event_Category'] = events_complete_subset['Single_Event'] + events_complete_subset['Team_Event']


# In[128]:


events_complete_subset.head()


# In[182]:


medal_tally_agnostic = events_complete_subset.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medals_Won', 'Event_Category']].agg('sum').reset_index()

medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medals_Won']/medal_tally_agnostic['Event_Category']
medal_tally_agnostic.sort_values(by='Medal_Won_Corrected').head()


# In[131]:


# Medal Tally.
medal_tally = medal_tally_agnostic.groupby(['Year','Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

medal_tally_pivot = pd.pivot_table(medal_tally,
                     index = 'Team',
                     columns = 'Year',
                     values = 'Medal_Won_Corrected',
                     aggfunc = 'sum',
                     margins = True).sort_values('All', ascending = False)[1:5]

# print total medals won in the given period
medal_tally_pivot.loc[:,'All']


# In[136]:


top_countries = ['USA', 'Russia', 'Germany', 'China']

year_team_medals = pd.pivot_table(medal_tally,
                                  index = 'Year',
                                  columns = 'Team',
                                  values = 'Medal_Won_Corrected',
                                  aggfunc = 'sum')[top_countries]

# plotting the medal tallies
year_team_medals.plot(linestyle = '-', marker = 'o', alpha = 0.9, figsize = (10,8), linewidth = 2)
plt.xlabel('Olympic Year')
plt.ylabel('Number of Medals')
plt.title('Olympic Performance Comparison')


# In[183]:


# List of top countries
top_countries = ['USA', 'Russia', 'Germany', 'China']

# row mask where countries match
row_mask_2 = medal_tally_agnostic['Team'].map(lambda x: x in top_countries)
row_mask_2.head()

# Pivot table to calculate sum of gold, silver and bronze medals for each country
medal_tally_specific = pd.pivot_table(medal_tally_agnostic[row_mask_2],
                                     index = ['Team'],
                                     columns = 'Medal',
                                     values = 'Medal_Won_Corrected',
                                     aggfunc = 'sum',
                                     fill_value = 0).drop('DNW', axis = 1)

# Re-order the columns so that they appear in order on the chart.
medal_tally_specific = medal_tally_specific.loc[:, ['Gold', 'Silver', 'Bronze']]

medal_tally_specific.plot(kind = 'bar', stacked = True, figsize = (8,6), rot = 0)
plt.xlabel('Number of Medals')
plt.ylabel('Country')


# In[151]:


best_team_sports = pd.pivot_table(medal_tally_agnostic[row_mask_2],
                                  index = ['Team', 'Event'],
                                  columns = 'Medal',
                                  values = 'Medal_Won_Corrected',
                                  aggfunc = 'sum',
                                  fill_value = 0).sort_values(['Team','Gold'],ascending=(True,False)).reset_index()
best_team_sports.drop(['Bronze','Silver','DNW'],axis=1,inplace=True)
best_team_sports.groupby('Team').head(5)


# In[169]:


# take for each year, the team, name of the athlete and gender of the athlete and drop duplicates. These are values
# where the same athlete is taking part in more than one sport.

# get rows with top countries
row_mask_3 = events_complete_subset['Team'].map(lambda x: x in top_countries)

year_team_gender = events_complete_subset.loc[row_mask_3, ['Year','Team', 'Name', 'Sex']].drop_duplicates()


# In[170]:


year_team_gender_count = pd.pivot_table(year_team_gender,
                                        index = ['Year', 'Team'],
                                        columns = 'Sex',
                                        aggfunc = 'count').reset_index()
year_team_gender_count.head()


# In[171]:


year_team_gender_count.columns = year_team_gender_count.columns.get_level_values(0)
year_team_gender_count.columns = ['Year', 'Team', 'Female_Athletes', 'Male_Athletes']
year_team_gender_count['Total Athletes']=year_team_gender_count['Female_Athletes']+year_team_gender_count['Male_Athletes']
year_team_gender_count



# In[172]:


# Separate country wise data

chi_data = year_team_gender_count[year_team_gender_count['Team'] == "China"]
chi_data.fillna(0, inplace = True)
chi_data.set_index('Year', inplace = True)

ger_data = year_team_gender_count[year_team_gender_count['Team'] == "Germany"]
ger_data.set_index('Year', inplace = True)

rus_data = year_team_gender_count[year_team_gender_count['Team'] == "Russia"]
rus_data.set_index('Year', inplace = True)

usa_data = year_team_gender_count[year_team_gender_count['Team'] == "USA"]
usa_data.set_index('Year', inplace = True)


# In[178]:


fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(20,12),sharey=True)
fig.subplots_adjust(hspace = 0.3)

ax1.bar(chi_data.index.values,chi_data['Male_Athletes'],align='edge',label='Male_Athletes')
ax1.bar(chi_data.index.values,chi_data['Female_Athletes'],align='edge',label='Female_Athletes')
ax1.plot(chi_data.index.values, chi_data['Total Athletes'], linestyle = ':', color = 'black', label = 'Total Athletes',
        marker = 'o')
ax1.set_title('Team China:\nComposition over the years')
ax1.set_ylabel('Number of Athletes')
ax1.legend(loc = 'best')

ax2.bar(ger_data.index.values,ger_data['Male_Athletes'],align='edge',label='Male_Athletes')
ax2.bar(ger_data.index.values,ger_data['Female_Athletes'],align='edge',label='Female_Athletes')
ax2.plot(ger_data.index.values, ger_data['Total Athletes'], linestyle = ':', color = 'black', label = 'Total Athletes',
        marker = 'o')
ax2.set_title('Team Germany:\nComposition over the years')
ax2.set_ylabel('Number of Athletes')
ax2.legend(loc = 'best')

ax3.bar(rus_data.index.values,rus_data['Male_Athletes'],align='edge',label='Male_Athletes')
ax3.bar(rus_data.index.values,rus_data['Female_Athletes'],align='edge',label='Female_Athletes')
ax3.plot(rus_data.index.values, rus_data['Total Athletes'], linestyle = ':', color = 'black', label = 'Total Athletes',
        marker = 'o')
ax3.set_title('Team Germany:\nComposition over the years')
ax3.set_ylabel('Number of Athletes')
ax3.legend(loc = 'best')

ax4.bar(usa_data.index.values,usa_data['Male_Athletes'],align='edge',label='Male_Athletes')
ax4.bar(usa_data.index.values,usa_data['Female_Athletes'],align='edge',label='Female_Athletes')
ax4.plot(usa_data.index.values, usa_data['Total Athletes'], linestyle = ':', color = 'black', label = 'Total Athletes',
        marker = 'o')
ax4.set_title('Team USA:\nComposition over the years')
ax4.set_ylabel('Number of Athletes')
ax4.legend(loc = 'best')

plt.show()


# In[184]:


# Get year wise team wise athletes.
year_team_athelete = events_complete_subset.loc[row_mask_3, ['Year','Team', 'Name']].drop_duplicates()

# sum these up to get total contingent size.
contingent_size = pd.pivot_table(year_team_athelete,
                                 index = 'Year',
                                 columns = 'Team',
                                 values = 'Name',
                                 aggfunc = 'count')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2,
                                         ncols = 2,
                                        figsize = (20,12))

fig.subplots_adjust(hspace = 0.3)

# Plot australia's medal tally and contingent size
contingent_size['China'].plot(ax = ax1, linestyle = '-', marker = 'o', linewidth = 2, color = 'red', 
                                  label = 'Contingent Size')
year_team_medals['China'].plot(ax = ax1, linestyle = '-', marker = 'o', linewidth = 2, color = 'black',
                                  label = 'Medal Tally')
ax1.plot(2008, contingent_size.loc[2008, 'China'], marker = '^', color = 'red', ms = 14)
ax1.plot(2008, year_team_medals.loc[2008, 'China'], marker = '^', color = 'black', ms = 14)
ax1.set_xlabel('Olympic Year')
ax1.set_ylabel('Number of Athletes/Medal Tally')
ax1.set_title('Team China\nContingent Size vs Medal Tally')
ax1.legend(loc = 'best')

# Plot USA's medal tally and contingent size
contingent_size['USA'].plot(ax = ax2, linestyle = '-', marker = 'o', linewidth = 2, color = 'blue',
                           label = 'Contingent Size')
year_team_medals['USA'].plot(ax = ax2, linestyle = '-', marker = 'o', linewidth = 2, color = 'black',
                            label = 'Medal Tally')
ax2.plot(1984, contingent_size.loc[1984, 'USA'], marker = '^', color = 'blue', ms = 14)
ax2.plot(1984, year_team_medals.loc[1984, 'USA'], marker = '^', color = 'black', ms = 14)
ax2.set_xlabel('Olympic Year')
ax2.set_ylabel('Number of Athletes/Medal Tally')
ax2.set_title('Team USA\nContingent Size vs Medal Tally')
ax2.legend(loc = 'best')

# Plot Germany's medal tally and contingent size
contingent_size['Germany'].plot(ax = ax3, linestyle = '-', marker = 'o', linewidth = 2, color = 'green',
                               label = 'Contingent Size')
year_team_medals['Germany'].plot(ax = ax3, linestyle = '-', marker = 'o', linewidth = 2, color = 'black',
                                label = 'Medal Tally')
ax3.plot(1972, year_team_medals.loc[1972, 'Germany'], marker = '^', color = 'black', ms = 14)
ax3.plot(1972, contingent_size.loc[1972, 'Germany'], marker = '^', color = 'green', ms = 14)
ax3.set_xlabel('Olympic Year')
ax3.set_ylabel('Number of Athletes/Medal Tally')
ax3.set_title('Team Germany\nContingent Size vs Medal Tally')
ax3.legend(loc = 'best')

# Plot Russia's medal tally and contingent size
contingent_size['Russia'].plot(ax = ax4, linestyle = '-', marker = 'o', linewidth = 2, color = 'orange',
                              label = 'Contingent Size')
year_team_medals['Russia'].plot(ax = ax4, linestyle = '-', marker = 'o', linewidth = 2, color = 'black',
                               label = 'Medal Tally')
ax4.plot(1980, contingent_size.loc[1980, 'Russia'], marker = '^', color = 'orange', ms = 14)
ax4.plot(1980, year_team_medals.loc[1980, 'Russia'], marker = '^', color = 'black', ms = 14)
ax4.set_xlabel('Olympic Year')
ax4.set_ylabel('Number of Athletes/Medal Tally')
ax4.set_title('Team Russia\nContingent Size vs Medal Tally')
ax4.legend(loc = 'best')

plt.show()

