import pandas as pd

def load_and_filter_data(file_path, text_column):
    """
    Load the dataset from a CSV file and filter rows where the text column contains text.
    
    Args:
    file_path (str): Path to the CSV file.
    text_column (str): Name of the column containing text comments.
    
    Returns:
    pd.DataFrame: Filtered DataFrame with non-empty text comments.
    """
    data = pd.read_csv(file_path)
    filtered_data = data[data[text_column].notna() & (data[text_column] != '')]
    return filtered_data

# Paths to your dataset files
file_path1 = 'dataset_tiktok-comments-scraper_2024-04-28_23-16-10-409.csv'
file_path2 = 'dataset_free-tiktok-scraper_2024-04-28_21-22-00-488.csv'

# Load and filter datasets
dataset1 = load_and_filter_data(file_path1, 'text')  # Assuming 'text' is the column for Dataset 1
dataset2 = load_and_filter_data(file_path2, 'text')  # Update 'text' if a different column name for Dataset 2

# Read the CSV files
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# # Display the first few rows of the filtered datasets
# print("Filtered Dataset 1:")
# print(dataset1.head())
# print("\nFiltered Dataset 2:")
# print(dataset2.head())

print("Column names in Dataset 1:")
print(dataset1.columns.tolist())  # Convert to list for easier viewing

print("\nColumn names in Dataset 2:")
print(dataset2.columns.tolist())



