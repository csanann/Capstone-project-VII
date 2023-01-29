# Capstone-project-VII
>>Click on RAW to see the code better
>>
my first Capstone project VII_draft on the UsArrests.csv
####Loading dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("UsArrests.csv")
df.head()

 ####Exploroatory Data Analysis (EDA) and preprocessing the dataset
df.info()
print(df.shape)

# get the percetage of null values in each column
df.isnull().sum() * 100 / df.shape [0]
# to upderstand the diversity of values in that feature
#call the numbers of features/types out
#to understand the distrition of the values in each feature
#..size file
unic_en = df.nunique()
print(unic_en)

# find entries = 1 unique entry
feat = unic_en[unic_en == 1].index
feat = feat.tolist()

print("\nFeature that has 1 unique entry:", feat)
# Getting a basic statistic out
df.describe()

#### The UsArrests dataset has 50 rows and 5 columns, including City is a object datatype, Murder and Rape are float, Assault and UrbanPop are numerical data. 
#### The dataset has no missing values and none has unique entry equal 1.

# to visualise representation of the distribution 
#and spread of each variable in the dataset, we need to import libaries
import matplotlib.pyplot as plt
import seaborn as sns

#Checking the correclation between variables in the dataset before narrow down to specific graph
# Compute the correlation matrix
corr = df.corr()

# Plot the correlation heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

### The correlation between the Assault and Murder variales in the UsArrests dataset is 0.8. This indicates that  there is a strong positive correlation between these two variables. The values of the Assault variable increase, means the values of the Murder variable are also likely to go up.
### While the correlation between the Assault and Rape variables in the same dataset is 0.67. This indicates that there is a moderate positive correlation between these two variables. It means that as the Assault values increase, the values of the Rap values are also increase.

# Plot histograms of each variable
df.hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.show()

The distribution on each variable is not normally distributed because it has multiple spikes. Therefore, the data has a complex distrition and or the data is spread out. We may need to transform the data to make it more normally distributed.

# Plot scatter plot of the relationship between Assault and Murder
plt.scatter(x=df['Assault'], y=df['Murder'])
plt.xlabel('Assault')
plt.ylabel('Murder')
plt.title('Scatter Plot of Assault and Murder')
plt.show()

# Plot scatter plot of the relationship between Assault and Rape
plt.scatter(x=df['Assault'], y=df['Rape'])
plt.xlabel('Assault')
plt.ylabel('Rape')
plt.title('Scatter Plot of Assault and Rape')
plt.show()

#### The scatterplot shows the postive relations between the two variables and clearly the Assault and Murder indicated the strong link than the Assault and Rape

# Plot box plot to check to visualise the relationship between variables and to check for outliers
sns.boxplot(data=df)
plt.show()

### Principal Composent Analysis (PCA), preprocessing

#double check before performing PCA
print(df.isnull().sum())

from sklearn.decomposition import PCA

df = pd.read_csv("UsArrests.csv")
#the model is initialised with n_components=2
#it will keep 2 new variables that capture the most information from the original variables
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.drop(["City"], axis=1))
#the df_pca will stored with the transformed data
df_pca = pd.DataFrame(df_pca, columns=["PC1","PC2"])
df_pca["City"] = df["City"]

print(df_pca)

#double check for missing values
print(df_pca.isnull().sum())

#to calculates the Pearson correlation matrix for the transformed data
print(df_pca.corr())

#### The correlation matrix shows the linear relationship between the two new variables, which is PC1 and PC2. PC1+PC1 has correlation between itself of 1.000000e+00 score and it means a positive linear relationship. While PC1 and PC2 are with correlation of  1.641348e-16, which shows weak linear relationship between the two variables. 

#### The finding shows the PC1 captures a significant number of information fromt he original variables, while PC2 captures very less information from the original variables.

#### Moreover, it can explain the PC1 represents a single linear combination of the variables and the PC2 as a residual component , captures information that not captured by PC1. To retain most of the information from the original data, only PC1 needs to be retained.

#calculates the standard deviation of each principal component in df_pca
#to see how much their values are spread out from their mean values
std = df_pca.describe().transpose()["std"]
print(f"Standard deviation: {std.values}")

#the explained variance ratio of each component
#it gives the contribution of each component to the total variability of the data
#where the sum of the explained variances is equal to 1
print(f"Proportion of Variance Explained: {pca.explained_variance_ratio_}")

#the cumulative proportion of the variance explained by each component
print(f"Cumulative Proportion: {np.cumsum(pca.explained_variance_)}")
It shows the first two components together explain over 98% of the variablility in the data.

# Standardize the features
X = (df - df.mean()) / df.std()

# Apply PCA to the standardized features
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X)

# Plot the biplot
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    fig, ax = plt.subplots()
    ax.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        ax.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is None:
            ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

biplot(df_pca, np.transpose(pca.components_), list(df.columns))
plt.show()

#### Each point represents one state/city in the UsArrests dataset, and the arrows represent the weight of each feature in the formation of the two principal components. The finding shows that there is a moderate amount of correlation betweenth e features but not a strong relationship. However, it is still contributing to the variation in the dat along both dimensions.

#to calculate the linkage matrix that stores the information about the distances between the data points
import scipy.cluster.hierarchy as shc

# Preprocess the data
#select only numerical columns for clustering
X = df.iloc[:,1:].values

# Perform Hierarchical Clustering
plt.figure(figsize=(10, 7))  
#to plot the dendrogram that represents the hierarchical relationship between the dta points
dend = shc.dendrogram(shc.linkage(X, method='ward'))
#ward method to minimizes the variance of the distance betweenthe clusters being merged
plt.title("Dendrogram")
plt.xlabel("States/Cities")
plt.ylabel("Euclidean distances")
plt.show()

#### The first set of long vertical lines, near the y-axis represent a high level of dissimilarity betweent he initial clusters. The second set of long vertical lines, further from the y-axis, represent a lower level of dissimilarity as the distance betweent he lines is shorter.

#### The finding indicates that the data points within these latter clusters are more similar to each other. the branching of the dendrogram and the shorter vertical lines within these branches further refine the structure of the clusters, dividing the data points into smaller ones and more  homogeneous groups.

#import the KMeans class 
from sklearn.cluster import KMeans

# initialise an instance of the KMeans class
#set to 3, Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the k-means algorithm on scaled data
kmeans.fit(df_pca)
# Predicting the clusters
pred_clusters = kmeans.predict(df_pca)

# Plotting the cluster assignments and cluster centers
plt.scatter(df_pca[:,0], df_pca[:,1], c=pred_clusters)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', s=200, linewidths=3, color='r')
plt.show()

#### The finding shows that the data has been separated ito 3 clusters based on their patterns of similarity. The K-Means algorithm grouped the dat points into 3 groups that represent with 3 stars. Each star represent the centerr of a cluster. The clusters are differentiated based on their location in the PCA transformed space.

















