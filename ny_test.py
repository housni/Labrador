# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print('Housni''s first love - Prasanthi')
x = [3.2,5]
print(type(x))


#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Load the data into a `pandas` DataFrame object
tr_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\member_sample.csv'
AAA_df = pd.read_csv(tr_path)

# Examine head of df
print(AAA_df.head(7))




#ASSIGNMENT 5 - Machine Learning

### This helper function will return an instance of a `DecisionTreeClassifier` with
### our specifications - split on entropy, and grown to depth of 1.
from sklearn.tree import DecisionTreeClassifier

def simple_tree():
    return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)


### Our example dataset, inspired from lecture
pts = [[.5, 3,1],[1,2,1],[3,.5,-1],[2,3,-1],[3,4,1],
 [3.5,2.5,-1],[3.6,4.7,1],[4,4.2,1],[4.5,2,-1],[4.7,4.5,-1]]

df = pd.DataFrame(pts, columns = ['x','y','classification'])

# Plotting by category

b = df[df.classification ==1]
r = df[df.classification ==-1]
plt.figure(figsize = (4,4))
plt.scatter(b.x, b.y, color = 'b', marker="+", s = 400)
plt.scatter(r.x, r.y, color = 'r', marker = "o", s = 400)
plt.title("Categories Denoted by Color/Shape")
plt.show()


print("df:\n",df, "\n")

### split out X and y
X = df[['x','y']]

# Change from -1 and 1 to 0 and 1
y = np.array([1 if x == 1 else 0 for x in df['classification']])

### Split data in half
X1 = X.iloc[:len(X.index)//2, :]
X2 = X.iloc[len(X.index)//2:, :]

y1 = y[:len(y)//2]
y2 = y[len(X)//2:]


### Fit classifier to both sets of data, save to dictionary:

tree_dict = {}

tree1 = simple_tree()
tree1.fit(X1,y1)
print("threshold:", tree1.tree_.threshold[0], "feature:", tree1.tree_.feature[0])

### made up alpha, for example
alpha1 = .6
tree_dict[1] = (tree1, alpha1)

tree2 = simple_tree()
tree2.fit(X2,y2)
print("threshold:", tree2.tree_.threshold[0], "feature:" ,tree2.tree_.feature[0])

### made up alpha, again.
alpha2 = .35

tree_dict[2] = (tree2, alpha2)

### Create predictions using trees stored in dictionary
print("\ntree1 predictions on all elements:", tree_dict[1][0].predict(X))
print("tree2 predictions on all elements:", tree_dict[2][0].predict(X))




## ASSIGNMENT 8 - Machine Learning

from scipy.stats import multivariate_normal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline

# File Paths
mv_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A8\mv.csv'
unif_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A8\unif.csv'
mv2_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A8\mv2.csv'
mv3_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A8\mv3.csv'

# Read in Data
mv_df = pd.read_csv(mv_path, index_col = 0)
unif_df = pd.read_csv(unif_path, index_col = 0)
mv2_df = pd.read_csv(mv2_path, index_col = 0)
mv3_df = pd.read_csv(mv3_path, index_col = 0)

# Create Figure
fig, (axs) = plt.subplots(2,2, figsize = (6,6))

# Plot each group in each dataset as unique olor
for ax, df in zip(axs.flatten(), [mv_df, unif_df, mv2_df, mv3_df]):
    for cat, col in zip(df['cat'].unique(), ["#1b9e77", "#d95f02", "#7570b3"]):
        ax.scatter(df[df.cat == cat].x, df[df.cat == cat].y, c = col, label = cat, alpha = .15)
    ax.legend()



## ASSIGNMENT 2 - Machine Learning
### This cell imports the necessary modules and sets a few plotting parameters for display

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)  

### Read in the data
tr_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A2\train.csv'
test_path = r'C:\Users\us0nya\Documents\PGDMLAI\Data\ML_A2\test.csv'

data = pd.read_csv(tr_path)  

### The .head() function shows the first few lines of data for perspecitve
data.head()
data.plot('GrLivArea', 'SalePrice', kind = 'scatter', marker = 'x');


