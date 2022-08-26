# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# %%

#read the files into a dataframe
df_hour=pd.read_csv('D:\downloads\hour.csv')
df_day=pd.read_csv ('D:\downloads\day.csv')


# %%
#view the 5 rows of the dataframe
df_hour.head(5)

# %%
#view first 5 rows of the dataframe
df_day.head(5)

# %%
#inspect the dataframes shape 
print(f'df_day shape:{df_day.shape}')
print(f'df_hour shape:{df_hour.shape}')
#null entries
print(df_day.isnull().sum())
print(df_hour.isnull().sum())

# %%
#duplicates 
print(df_day.duplicated().sum())
df_hour.duplicated().sum()

# %%
#hour dataframe describe stat
df_hour.describe().T

# %%
#day dataframe describe stat, T character after the describe() method gets the transpose of the 
#resulting dataset, hence the columns become rows and vice versa.
df_day.describe().T

# %% [markdown]
# ### Preprocessing Temporal and Weather Features

# %%
# create a copy of the original data
hour=df_hour.copy()
day=df_day.copy()


# %%
hour['season'].unique()

# %%
#change the seasons to Winter, summer, spring and fall
seasons= {1:'winter',2:'spring',3:'summer',4:'fall'}
hour['season']=hour['season'].apply(lambda x: seasons[x])
#view the change
hour['season'].unique()

# %%
hour['yr'].unique()

# %%
years={0:2011,1:2012}
hour['yr']=hour['yr'].apply(lambda x: years[x])
hour['yr'].unique()

# %%
hour['weekday'].unique()

# %%
weekdays={0: 'Sunday', 1: 'Monday', 2: 'Tuesday',3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
hour['weekday']=hour['weekday'].apply(lambda x: weekdays[x])
hour['weekday'].unique()

# %%
#tranform weather 
weather= {1:'clear',2:"cloudy",3:'light_rain_snow',4:'heavy_rain_snow'}
hour['weathersit']= hour['weathersit'].apply(lambda x: weather[x])
hour['weathersit'].unique()

# %%
#transform humidity and windspeed
hour['hum']=hour['hum']*100
hour['windspeed']=hour['windspeed']*67


# %%
#view the changes on the columns changed 
hour.sample(10,random_state=123)

# %%
hour.head(5)

# %%
#check that the number of the registered and casuals is eqaul to all the rides counted 
assert(hour.registered+hour.casual==hour.cnt).all(),'Sum of casual and registered rides not equal ''to total number of rides'

# %% [markdown]
# #### DISTRIBUTION OF RIDES (CASUAL , REGISTERED)

# %%
#create a function for xlabel , ylabel and title
def x_y_t(xlabel,ylabel,title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# %%
#distribution of registered vs casual rides
sb.distplot(hour['registered'],label='registered')
sb.distplot(hour['casual'],label='casual')
x_y_t('rides','Frequency','Rides Distribution')
plt.legend()


# %% [markdown]
# ### Plot observations
# > The plot above shows that the Registered users have more rides as compared to the casual users.
# 
# > the two distributions are skewed to the right, meaning that most of the entries in the data , zero or a small number of rides were registerd 
# 
# > Finally, every entry in the data has quite a large number of rides (that is, higher than 800).

# %% [markdown]
# #### EVOLUTION OF RIDES OVER TIME

# %%
#evolution of rides over time
plot_data = hour[['registered', 'casual', 'dteday']]
ax = plot_data.groupby('dteday').sum().plot(figsize=(12,8))
x_y_t('time','rides per day','Evolution of Rides over time')



# %% [markdown]
# ### Plot observations
# 
# > the number of registered rides is always above and significantly higher than the number of casual rides per day.
# 
# > During winter, the overall number of rides decreases 

# %%
#create a new dataframe with the columns to plot
plot_df=hour[['registered','casual','dteday']]
#group by the dteday column
plot_df=plot_df.groupby('dteday').sum()
#define window for computing the rolling mean and standard deviation
window=7
#rolling window function lets you calculate new values over each row in a dataframe
#to compute the rolling statistics , i.e the mean and the standard deviation, use the rolling function in which mean() adn std( ) is used to compute the rolling mean and rolling standard deviation

#the value of the rolling mean or the standard deviation at a certain time instance is only computed from the last window ebtries in the time series
# in this case, the window is 7 , and not from the entries of the whole series

rolling_mean=plot_df.rolling(window).mean()
rolling_deviation=plot_df.rolling(window).std()
#Create a plot of the series, where we first plot the series of rolling 
#means, then we color the zone between the series of rolling means +- 2 
#rolling standard deviations

#create a figure to plot on
ax = rolling_mean.plot(figsize=(12,8))

#plot : fill_between() is used to fill area between two horizontal curves. that is the registered curve and the casual curve
ax.fill_between(rolling_mean.index,rolling_mean['registered']+2*rolling_deviation['registered'],rolling_mean['registered']-2*rolling_deviation['registered'],alpha=0.2)


ax.fill_between(rolling_mean.index, rolling_mean['casual'] + 2*rolling_deviation['casual'], rolling_mean['casual'] - 2*rolling_deviation['casual'],alpha = 0.2)
#x_y_t function
x_y_t('time','Number of rides per day','Rides aggregated')


# %%


# %% [markdown]
# #### Rides distrubution through the hours of the day

# %%
# create a dataframe for the relevant columns
plot_hour= hour[['hr','registered','casual','weekday']]
#melt function is used to change the DataFrame format from wide to long
#It's used to create a specific format of the DataFrame object where one or more columns work as identifiers. 
#All the remaining columns are treated as values and unpivoted to the row axis and only two columns - variable and value
plot_hour=plot_hour.melt(id_vars=['hr','weekday'],var_name='type',value_name='count')
#create a facet grid
#the seaborn.FacetGrid() creates a new grid of plots, with rows corresponding to the different days of the week and columns responding to the types
hour_grid=sb.FacetGrid(plot_hour,row='weekday',col='type',height=2.5,aspect=2.5,row_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
#map creates the respective plots 
hour_grid.map(sb.barplot,'hr','count',alpha=0.5)


# %%
# above plot function
# the melt function() on the dataset will create a new dataset in which values are grouped by the hr and weekday columns,while creating two new columns : type ( which contains casual and # # # registered values) and  count (containing respective counts for the casual and registered types)
#view the melted dataframe
plot_hour.sample(10)

# %% [markdown]
# ### Plot Observations
# > On working days , the highest number of rides for registered users takes place around 8 AM and 6 PM. This mostly indicated that the registered users use the rides for commuting.
# 
# > The casual usage of bike sharing services on working days is quite limited.
# 
# > During the weekend, the ride distribution change for both casual and registered users . The registered rides are still more frequent than casual ones but bith distribution have the same shape, almost uniformly distributed between 11 Am and 6 PM.
# 
# > Most of the usage of bike sharing services occurs during working days, right before and right after the standard working time (that is, 9 to 5)

# %% [markdown]
# ## Seasonal Impact on Rides

# %%
#create a dataframe with specific relevant columns
plot_df= hour[['hr','season','registered','casual']]
#unpivot the data from wide to long 
plot_df= plot_df.melt(id_vars=['hr','season'],var_name='type',value_name='count')
#define the facet grid
grid= sb.FacetGrid(plot_df,row='season',col='type',height=2.5,aspect=2.5,row_order=['winter','spring','summer','fall'])
# map the grid using the seaborn barplot
grid.map(sb.barplot, 'hr', 'count', alpha=0.5)

# %% [markdown]
# ### Plot Observation
# > There are fewer rides in winter
# 
# > The distributions shapes are similar through th seasons

# %%
#create a dataframe with weekday, seasons , registered and casual
df_plot=hour[['weekday','season','registered','casual']]
#unpivot the data to long data format
df_plot=df_plot.melt(id_vars={'weekday','season'},var_name='type',value_name='count')
#define facetgrid
grid=sb.FacetGrid(df_plot,row='season',col='type',height=2.5,aspect=2.5,row_order=['winter','spring','summer','fall'])
#plot the grid using the seaborn barplot
grid.map(sb.barplot,'weekday','count',alpha=0.5,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])



# %% [markdown]
# ### Plot Observation
# >  There is a decreasing number of registered rides over the weekend as compared to the rest of the week , while the number of casual rides increases.
# 
# > This could enforce the initial hypothesis , that the registered customers mostly use the bike sharing service for commuting ,which coulc explain why the number decreases over the weekend,while the casual customers use the serviced occasionally over the weekend.
# 

# %% [markdown]
# ## Hypothesis Tests

# %%
#compute the population mean of the registered 
registered_mean=hour['registered'].mean()
# get sample of the data (summer 2011)\
registered_sample= hour[(hour.season=='summer') & (hour.yr==2011)].registered
#perform t-test and compute p-value
from scipy.stats import ttest_1samp
test_result = ttest_1samp(registered_sample, registered_mean)
print(f"Test statistic: {test_result[0]}, \
p-value: {test_result[1]}")

# %%
registered_mean=hour['registered'].mean()
# get sample as 5% of the full data
import random
random.seed(111)
sample_unbiased = hour.registered.sample(frac=0.05)
test_result_unbiased = ttest_1samp(sample_unbiased, registered_mean)
print(f"Unbiased test statistic: {test_result_unbiased[0]}, \
p-value: {test_result_unbiased[1]}")

# %% [markdown]
# Null hypothesis : H_0 : average registered rides over weekdays-average registered rides over 
# weekend=0
# 
# and 
# 
# alternative hypothesis : H_a : average registered rides over weekdays-average registered rides over 
# weekend≠0

# %%
#define mask,indicating if the day is weekend or work day
weekend_days= ['Saturday','Sunday']
weekend=hour.weekday.isin(weekend_days)
work_days = ~hour.weekday.isin(weekend_days)
#select registered rides for the weekend and working days
weekend_data= hour.registered[weekend]
work_days = hour.registered[work_days]
# perform ttest
from scipy.stats import ttest_ind


test_res = ttest_ind(weekend_data, work_days)
print(f"Statistic value: {test_res[0]:.03f}, \
p-value: {test_res[1]:.03f}")

# %%
# define mask, indicating if the day is weekend or work day
weekend_days = ['Saturday', 'Sunday']
weekend_mask = hour.weekday.isin(weekend_days)
workingdays_mask = ~hour.weekday.isin(weekend_days)

# select registered rides for the weekend and working days
weekend_data = hour.registered[weekend_mask]
workingdays_data = hour.registered[workingdays_mask]

# perform ttest
from scipy.stats import ttest_ind
test_res = ttest_ind(weekend_data, workingdays_data)
print(f"Statistic value: {test_res[0]:.03f}, p-value: {test_res[1]:.03f}")

# plot distributions of registered rides for working vs weekend days
sb.distplot(weekend_data, label='weekend days')
sb.distplot(workingdays_data, label='working days')
plt.legend()
plt.xlabel('rides')
plt.ylabel('frequency')
plt.title("Registered rides distributions")

# %%
# select casual rides for the weekend and working days
weekend_data = hour.casual[weekend_mask]
workingdays_data = hour.casual[workingdays_mask]

# perform ttest
test_res = ttest_ind(weekend_data, workingdays_data)
print(f"Statistic value: {test_res[0]:.03f}, p-value: {test_res[1]:.03f}")

# plot distributions of casual rides for working vs weekend days
sb.distplot(weekend_data, label='weekend days')
sb.distplot(workingdays_data, label='working days')
plt.legend()
x_y_t('Ride','Frequency','Casual rides distribution')


# %% [markdown]
# ## Weather-Related Features
# 
# > Expecting to observe a strong dependency of those features on the current number of rides , as bad weather can significantly influence bike sharing services .
# 
# > The weather features we identified earlier are the following: weatherit(Clear , cloudy, light_rain_snow, heavy_rain _snow), temp(normalized temperatures in celcius), atemp(normalized feeling temperature in celcius), hum( humidity level as a percentage), windspeed (wind speed in m/s)
# 
# 

# %%
#create a correlation function to avoid repetition 
def correlation_plt (data,col):
#get correlation between col and registered rides
    corr_registered= np.corrcoef(data[col],data['registered'])[0,1]
    #define the plot : seaborn regplot, col is the column to be plot eg temp, hum
    ax= sb.regplot(x=col,y='registered',data=data,scatter_kws={"alpha":0.05},label=f'registered rides(correlation:{corr_registered:.3f}')
#get correlation between col and casual rides
    corr_casual= np.corrcoef(data[col],data['casual'])[0,1]
    ax=sb.regplot(x=col,y='casual',data=data,scatter_kws={'alpha':0.05},label=f'casual rides (correlation: {corr_casual:.3f}')
#adjust legend alpha
    legend=ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(0.5)
    ax.set_ylabel("rides")
    ax.set_title(f"Correlation between rides and {col}")
    return ax

# %%
#plot the correlation for temp column
plt.figure(figsize=[10,6])
ax=correlation_plt(hour,'temp')


# %% [markdown]
# ### Plot Observation 
# > The correlation between the rides and temp for casual rides is 0.460 and the registered rides is 0.335
# 
# > There is a positive correlation between the number of the rides and temperature

# %%
#plot the correlation between the rides and atemp
plt.figure(figsize=[10,6])
ax= correlation_plt(hour,'atemp')


# %% [markdown]
# ### Plot Observation
# 
# > The correlation of the number of rides and atemp is closely similar to the temperatures correlation

# %%
#plot the correlation between the rides and the humidity
plt.figure(figsize=[10,6])
ax=correlation_plt(hour,'hum')

# %% [markdown]
# ### Plot Observation
# 
# > There is a negative correlation for both the registered rides and casual rides, which means that with high level of humidity , customers will tend no to use the bike sharing service. 
# 

# %%
#plot the correlation between the rides and windspeed
plt.figure(figsize=[10,6])
ax= correlation_plt(hour,'windspeed')


# %% [markdown]
# ### Plot observation
# > There is minimal correlation between the number of rides and wind speed. 
# 
# > A correlation close to 0 is referred to a weak correlation. In this plot the correlation is a weeak positive correlation
# 
# 

# %% [markdown]
# ## The Difference between the Pearson and Spearman Correlations
# 
# > Pearson correlation assumes a linea relationship between the two variables
# 
# > SPearman correlation only requires a monotonic relationship.
# 
# 

# %%
#define random variables , create an X variable which will represent the independent variable, and two dependent ones , Ylin an Ymon
x=np.linspace(0,5,100)

y_lin=0.5*x +0.1*np.random.randn(100)
y_mon = np.exp(x) +0.1 *np.random.randn(100)



# %%
#compute the pearson and spearman correlation using the pearsonr() adn spearmanr() functions in the scipy.stats module
from scipy.stats import pearsonr , spearmanr
corr_lin_pearson =pearsonr(x,y_lin)[0]
corr_lin_spearman = spearmanr(x,y_lin)[0]
corr_mon_pearson = pearsonr(x,y_mon)[0]
corr_mon_spearman = spearmanr(x,y_mon)[0]

#both pearsonr () and spearmanr() functions return a two dimensional arry in which the first value is the correlation and second value id p_value of the hypothesis test in which the 
# null hypothesis assumes that the computed correlation is equal to zero.
# the above code asks for the first value of the array indexed as [0]
print(f'corr_mon_spearman: {corr_mon_spearman}, corr_lin_spearman: {corr_lin_spearman},corr_lin_pearsonr: {corr_lin_pearson},corr_mon_pearsonr:{corr_mon_pearson}')

# %%
#Visualize both the data and the computed correlations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.scatter(x, y_lin)
ax1.set_title(f"Linear relationship\n \
Pearson: {corr_lin_pearson:.3f}, \
Spearman: {corr_lin_spearman:.3f}")
ax2.scatter(x, y_mon)
ax2.set_title(f"Monotonic relationship\n \
Pearson: {corr_mon_pearson:.3f}, \
Spearman: {corr_mon_spearman:.3f}")

# %% [markdown]
# ### Plot Observation 
# > On the first plot, the relation is linear and the two correlation coefficients are very similar.
# 
# > On the second plot , the monotonic relationship, the linear assumption of the pearson correlation fails, although the correlation is still high, (not capable of capturinf the perfect relationship  between the two variables) 
# 
# > On the monotonic relationship plot, the spearman correlation coefficient is 1, meaning that it succeeds in captuting the almost perfect relationship between the two variables
# 
# 

# %%
#define a function , computing the pearson and spearman correlation coefficients with registered and casual rides in the hour dataset
def correlation_fn(data,col):
    pearson_registered = pearsonr(data[col], data["registered"])[0]
    pearson_casual = pearsonr(data[col],data['casual'])[0]
    spearman_registered = spearmanr(data[col],data['registered'])[0]
    spearman_casual = spearmanr(data[col],data['casual'])[0]
    return pd.Series({'Pearson (registered)': pearson_registered,'Spearman (registered)': spearman_registered,'Pearson (casual)':pearson_casual,'Spearman (casual)':spearman_casual})

# the defined correlation_fn function returns a pandas.Seried () object which will be used to create a new dataset containing the different correlations


# %%
#compute correlation measures between different features 
cols= ['temp','atemp','hum','windspeed']
corr_data= pd.DataFrame(index=['Pearson (registered)','Spearman (registered)','Pearson (casual)','Spearman (casual)'])
for col in cols:
    corr_data[col]=correlation_fn(hour,col)
corr_data.T


# %% [markdown]
# ### Table Observations
# > For most of the variables, the Pearson and Spearman correlation coefficient are close enough
# 
# > The most visible difference is between the casual columns focusing on temp and atemp. The spearman correlation is higher which means that there is a significant evidence fr a non-linear,relatively strong and positive relationship.
# 
#  > This can be interpretated that casual customers are more keen on using the bike sharing service when temperatures are higher.
# 
# > Registered rides are not stronghly correlated to temp as the casual rides. Can coclude that the casual rides are used for the weekends and not majorly for commuting.

# %% [markdown]
# ## Correlation Matrix Plot

# %%
# plot correlation matrix
cols = ["temp", "atemp", "hum", "windspeed", "registered", "casual"]
plot_data = hour[cols]
corr = plot_data.corr()
fig = plt.figure(figsize=(10,8))
plt.matshow(corr, fignum=fig.number)
plt.xticks(range(len(plot_data.columns)), plot_data.columns)
plt.yticks(range(len(plot_data.columns)), plot_data.columns)
plt.colorbar()
plt.ylim([5.5, -0.5])

# %% [markdown]
# ## Time Series Analysis
# 
# >A time series is a sequence of observations equally spaced in time and in chronological order.

# %%
#define a function for plotting rolling stats and Augumented Dickey- Fuller for time series
from statsmodels.tsa.stattools import adfuller
def test_stationarity(ts,window=10,**kwargs):
    #create a dataframe for plotting
    plot_df= pd.DataFrame(ts)
    plot_df['rolling_mean']= ts.rolling(window).mean()
    plot_df['rolling_deviation']= ts.rolling(window).std()
    #compute the p value of dickey-fuller tests
    p_val= adfuller(ts)[1]
    ax=plot_df.plot(**kwargs)
    ax.set_title(f'Dickey-Fuller p-value:{p_val:.3f}')



# %%
#get daily rides
rides=hour[['dteday','registered','casual']]
rides= rides.groupby('dteday').sum()
#convert index to Dtaframe object
rides.index = pd.to_datetime(rides.index)

# %%
#registered
test_stationarity(rides.registered,figsize=(10,8))

# %%
#casual
test_stationarity(rides.casual,figsize=(10,8))

# %% [markdown]
# ### Plot Observation
#  > Neither the moving average nor standard deviations are statinary.
# 
# > The dickey-fuller tests results for the registered is 0.355 and the casual is 0.372. This is strong evidence that the time series is not stationary
# 
# > To detrend a time series and make it stationary : subtract either its rolling mean or its last value
# 

# %%
# subtract rolling mean
registered = rides["registered"]
registered_ma = registered.rolling(10).mean()
registered_ma_diff = registered - registered_ma
registered_ma_diff.dropna(inplace=True)
casual = rides["casual"]
casual_ma = casual.rolling(10).mean()
casual_ma_diff = casual - casual_ma
casual_ma_diff.dropna(inplace=True)

# %%
#registered
test_stationarity(registered_ma_diff,figsize=(10,6))

# %%
#casual
test_stationarity(casual_ma_diff,figsize=(10,6))

# %% [markdown]
# ###  Time Series Decomposition in Trend, Seasonality, and Residual Components

# %%
#Use the statsmodel.tsa.seasonal. seasonal_decompose() method to decompose the registered and casual rides:
from statsmodels.tsa.seasonal import seasonal_decompose
reg_decomposition= seasonal_decompose(rides['registered'])
cas_decomposition= seasonal_decompose(rides['casual'])
#plot decompositions
reg_plot= reg_decomposition.plot()
reg_plot.set_size_inches(10,8)
cas_plot= cas_decomposition.plot()
cas_plot.set_size_inches(10,8)

# %%
# test residuals for stationarity
plt.figure()
test_stationarity(reg_decomposition.resid.dropna(),figsize=(10, 8))
plt.figure()
test_stationarity(cas_decomposition.resid.dropna(),figsize=(10, 8))

# %% [markdown]
# ### ARIMA Models

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
fig, axes = plt.subplots(3, 3, figsize=(25, 12))
# plot original series
original = rides["registered"]
axes[0,0].plot(original)
axes[0,0].set_title("Original series")
plot_acf(original, ax=axes[0,1])
plot_pacf(original, ax=axes[0,2])
# plot first order integrated series
first_order_int = original.diff().dropna()
axes[1,0].plot(first_order_int)
axes[1,0].set_title("First order integrated")
plot_acf(first_order_int, ax=axes[1,1])
plot_pacf(first_order_int, ax=axes[1,2])
# plot first order integrated series
second_order_int = first_order_int.diff().dropna()
axes[2,0].plot(first_order_int)
axes[2,0].set_title("Second order integrated")
plot_acf(second_order_int, ax=axes[2,1])
plot_pacf(second_order_int, ax=axes[2,2])


# %%
conda install -c saravji pmdarima

# %%
from pmdarima import auto_arima
model = auto_arima(registered, start_p=1, start_q=1, max_p=3, max_q=3, information_criterion="aic")
print(model.summary())

# %%
# plot original and predicted values
plot_data = pd.DataFrame(registered)
plot_data['predicted'] = model.predict_in_sample()
plot_data.plot(figsize=(12, 8))
plt.ylabel("number of registered rides")
plt.title("Predicted vs actual number of rides")

# %% [markdown]
# 

# %% [markdown]
# 


