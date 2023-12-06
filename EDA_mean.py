import pandas as pd

# Assuming your CSV data is stored in a file named 'your_data.csv'
file_path = 'image_counts.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=',')

# Remove leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Calculate the mean for 'female', 'male', and 'total_by_ethinicity' columns after grouping by 'Age'
mean_images_per_age = df.groupby('Age')[['female', 'male', 'total_by_ethinicity']].mean()

# Calculate the overall mean of 'total_by_ethinicity' across all age groups
overall_mean_total_by_ethinicity = df['total_by_ethinicity'].mean()

# Print the results
print("Mean images per age:")
print(mean_images_per_age)

print("\nOverall mean total_by_ethinicity:")
print(overall_mean_total_by_ethinicity)
