import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

### DATA LOADING ###
df = pd.read_csv('../fraud_raw.csv')

### DATA PREPROCESSING ###
### Feature Engineering (adding days to expire)
df['Expiry'] = '01/' + df['Expiry']  # Now format is DD/MM/YY

# Parse with specified format
df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d/%m/%y', errors='coerce')

# Define the reference date (e.g., today)
today = pd.Timestamp.today()

# Calculate days remaining
df['DaysToExpire'] = (df['Expiry'] - today).dt.days

### One-hot Encoding
prof_data = df[['Profession']]

# Create encoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform
encoded_array = encoder.fit_transform(prof_data)

# Get new column names
encoded_cols = encoder.get_feature_names_out(['Profession'])

# Create DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

# Concatenate with original df (excluding original 'Profession')
df_encoded = pd.concat([df.drop(columns=['Profession']), encoded_df], axis=1)

### Standarization
scaler = StandardScaler()

# Select columns to standardize
data_to_scale = df_encoded[['Income', 'DaysToExpire']]

# Fit scaler and transform data
scaled_data = scaler.fit_transform(data_to_scale)

df_encoded[['Income', 'DaysToExpire']] = scaled_data

### Remove Unused Feature
df_encoded = df_encoded.drop(columns=['Credit_card_number', 'Security_code', 'Expiry'])

### Export to csv
df_encoded.to_csv('fraud_preprocessing.csv')