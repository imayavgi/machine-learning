import pandas as pd
import matplotlib.pyplot as plt

# Print the summary statistics
df = pd.read_csv('TrainingData.csv')
print(df.describe())

# Create the histogram
plt.hist(df['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()