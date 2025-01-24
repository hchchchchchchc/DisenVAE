import  pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train_rating.csv')
test_df = pd.read_csv('test_rating.csv')
valid_df = pd.read_csv('validate_rating.csv')

combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

conbined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = 0.7
valid_size = 0.15
test_size = 0.15

train_data, temp_data = train_test_split(conbined_df, train_size=train_size, random_state=42)

valid_data, test_data = train_test_split(temp_data, test_size=test_size / (test_size + valid_size), random_state=42)

train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)