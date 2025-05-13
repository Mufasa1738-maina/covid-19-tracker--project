
---

## COVID-19_Global_Data_Tracker.ipynb

```python
# %% [markdown]
# # COVID-19 Global Data Tracker
# 
# ## Project Overview
# This notebook analyzes global COVID-19 data including cases, deaths, and vaccination progress.

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## 1. Data Loading

# %%
# Load the dataset
try:
    df = pd.read_csv('data/owid-covid-data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: Data file not found. Please ensure 'owid-covid-data.csv' is in the data/ directory.")
    
# Display basic info
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
display(df.head())

# %%
# Show column information
print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# %% [markdown]
# ## 2. Data Cleaning

# %%
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for countries (exclude continent aggregates)
countries_to_include = ['United States', 'India', 'Brazil', 'United Kingdom', 'Germany', 'Kenya', 'South Africa']
df = df[df['location'].isin(countries_to_include)]

# Select relevant columns
columns_to_keep = [
    'date', 'location', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'population', 'population_density', 'median_age', 'gdp_per_capita'
]
df = df[columns_to_keep]

# Handle missing values
df['total_cases'] = df['total_cases'].fillna(0)
df['new_cases'] = df['new_cases'].fillna(0)
df['total_deaths'] = df['total_deaths'].fillna(0)
df['new_deaths'] = df['new_deaths'].fillna(0)

# Calculate derived metrics
df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases']) * 100
df['vaccination_rate'] = (df['people_vaccinated'] / df['population']) * 100

# Sort by date and location
df = df.sort_values(['location', 'date'])

# Display cleaned data
print("\nCleaned data shape:", df.shape)
display(df.head())

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
# Summary statistics
print("Summary statistics for numerical columns:")
display(df.describe())

# %%
# Time series of total cases
plt.figure(figsize=(14, 7))
for country in df['location'].unique():
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Cases (millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/total_cases_trend.png')
plt.show()

# %%
# New cases rolling average
plt.figure(figsize=(14, 7))
for country in df['location'].unique():
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], 
             country_data['new_cases'].rolling(7).mean(), 
             label=country)

plt.title('7-Day Moving Average of New COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('New Cases (7-day avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/new_cases_trend.png')
plt.show()

# %%
# Case fatality rate comparison
latest_data = df[df['date'] == df['date'].max()]
plt.figure(figsize=(12, 6))
sns.barplot(x='location', y='case_fatality_rate', data=latest_data)
plt.title('Case Fatality Rate by Country (Latest Data)')
plt.xlabel('Country')
plt.ylabel('Fatality Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/fatality_rate.png')
plt.show()

# %% [markdown]
# ## 4. Vaccination Analysis

# %%
# Vaccination progress
plt.figure(figsize=(14, 7))
for country in df['location'].unique():
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], 
             country_data['vaccination_rate'], 
             label=country)

plt.title('COVID-19 Vaccination Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Percentage of Population Vaccinated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/vaccination_progress.png')
plt.show()

# %%
# Latest vaccination status
latest_vaccines = latest_data.sort_values('people_fully_vaccinated', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='location', y='people_fully_vaccinated', data=latest_vaccines)
plt.title('Total Fully Vaccinated People by Country')
plt.xlabel('Country')
plt.ylabel('Fully Vaccinated Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/fully_vaccinated.png')
plt.show()

# %% [markdown]
# ## 5. Interactive Visualizations (Plotly)

# %%
# Interactive time series of cases
fig = px.line(df, x='date', y='total_cases', color='location',
              title='Interactive COVID-19 Cases Timeline',
              labels={'total_cases': 'Total Cases', 'date': 'Date'},
              template='plotly_white')
fig.show()

# %%
# Choropleth map (requires ISO codes)
try:
    # This assumes you have country ISO codes in your data
    fig = px.choropleth(latest_data,
                        locations="iso_code",
                        color="total_cases",
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Global COVID-19 Cases")
    fig.show()
except Exception as e:
    print(f"Could not create choropleth map: {e}")

# %% [markdown]
# ## 6. Key Insights

# %%
# Calculate and display insights
latest_date = df['date'].max()
total_cases_world = df.groupby('location')['total_cases'].max().sum() / 1e6
total_deaths_world = df.groupby('location')['total_deaths'].max().sum() / 1e3

print(f"\nKey Insights as of {latest_date.strftime('%B %d, %Y')}:")
print(f"1. Total cases across analyzed countries: {total_cases_world:.1f} million")
print(f"2. Total deaths across analyzed countries: {total_deaths_world:.1f} thousand")

# Find country with highest vaccination rate
most_vaccinated = latest_data.loc[latest_data['vaccination_rate'].idxmax()]
print(f"3. {most_vaccinated['location']} has the highest vaccination rate at {most_vaccinated['vaccination_rate']:.1f}%")

# Find country with highest fatality rate
highest_fatality = latest_data.loc[latest_data['case_fatality_rate'].idxmax()]
print(f"4. {highest_fatality['location']} has the highest case fatality rate at {highest_fatality['case_fatality_rate']:.1f}%")

# Calculate days since first case
first_cases = df[df['total_cases'] > 0].groupby('location')['date'].min()
for country, date in first_cases.items():
    days = (latest_date - date).days
    print(f"5. {country} has been reporting cases for {days} days")

# %% [markdown]
# ## 7. Exporting Results

# %%
# Save cleaned data
df.to_csv('data/covid_cleaned_data.csv', index=False)

# Save latest snapshot
latest_data.to_csv('data/covid_latest_snapshot.csv', index=False)

print("\nAnalysis complete! Data exports saved to data/ directory.")
