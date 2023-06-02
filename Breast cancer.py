#To conduct data interpretation, preprocessing and build machine learning algorithm, we employ a variety of Python packages mentioned below:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as warning
warning.filterwarnings('ignore')
from IPython.core.display import display, HTML
from IPython.display import display, HTML
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from scipy import stats
#Scaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
#Scoring & evalutating
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.model_selection import cross_validate
from IPython.display import display, HTML
#Libraries for Modeling
from configparser import ConfigParser
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression #LogisticRegression
from sklearn.model_selection import KFold #K-Fold
from keras.models import Sequential #Deep Learning
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D #Deep Learning
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict
#To conduct data interpretation, preprocessing and build machine learning algorithm, we employ a variety of Python packages mentioned below:

#Data processing and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as warning
warning.filterwarnings('ignore')
from IPython.core.display import display, HTML
from IPython.display import display, HTML
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from scipy import stats
#Scaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
#Scoring & evalutating
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.model_selection import cross_validate
from IPython.display import display, HTML
#Libraries for Modeling
from configparser import ConfigParser
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression #LogisticRegression
from sklearn.model_selection import KFold #K-Fold
from keras.models import Sequential #Deep Learning
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D #Deep Learning
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict


# In[2]:

#1. Data understanding

##1.1 Read the `breast-cancer.csv` file to a pandas dataframe, and then display using `cancer` for overview.
cancer = pd.read_csv('breast-cancer.csv')
cancer


# In[3]:


#1.2 `cancer.dtypes` attribute used to understand the structure of a dataframe in order to ensure that each column has the correct data type.

cancer.dtypes


# In[4]:


#1.3a To determine the total number of benign (B) and malignant (M) in diagnosis column, `cancer[''diagnosis''].value_counts()` function was used.

cancer['diagnosis'].value_counts()


# In[5]:


#1.3b To have a visual representation, we plotted the diagnosis counts using `px.bar`.

# Create the value_counts for the diagnosis column
value_counts = cancer['diagnosis'].value_counts()

# Create a bar plot using Plotly Express.
fig = px.bar(value_counts, x=value_counts.index, y='diagnosis', color='diagnosis', color_continuous_scale='GnBu')

# Update the title.
fig.update_layout(title='Diagnosis Counts', title_x=0.5)

# Show the plot.
fig.show()


# In[6]:


#1.4 Dividing the range of values into `bins` and plotting of each numerical column using `num_data.hist`.

fig, ax = plt.subplots(figsize=(20,15))
plot_data = cancer.select_dtypes(include=['float64'])
plot_data.hist(bins=50, ax=ax)
fig.suptitle("Histograms of Numerical Columns", fontsize=20)
plt.show()


# In[7]:


#1.5 Using `cancer.describe()` to user the statistics for each numerical column.

cancer.describe()


# In[8]:


#1.6 Using `sns.heatmap` to understand the relationship between two or more variables.

sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(cancer.corr(), cmap="YlGnBu", annot=False)


# In[9]:


#1.7 Using 'sns.swarmplot' to create swarmplot showing the distribution of values for each class of a binary variable for a specific column

# Specify the list of columns to plot
columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
           'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
           'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
           'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
           'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst']

# Set up the subplot grid
fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(18, 28))
axs = axs.flatten()

# Add more space between the subplots
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot swarmplots for each column in the list, with diagnosis as the hue
for i, column in enumerate(columns):
    # Plot the swarmplot for the current column, with diagnosis as the hue
    sns.swarmplot(x='diagnosis', y=column, data=cancer, ax=axs[i])
    # Add a title to the plot
    axs[i].set_title(column)


# ### 2. Data pre-processing

# In[10]:


#2.1 `cancer.isnull().sum().sum()`attribute used to determine the amount of missing values in a dataset.

cancer.isnull().sum().sum()


# In[11]:


#2.2 `cancer.drop_duplicates()`function used to determine if there is any missing values in a dataset and if there is, drop them.

cancer.drop_duplicates()


# In[12]:


#2.3 Dropping id colums using `cancer.drop`.

cancer = cancer.drop('id', axis=1)
cancer.head()


# In[13]:


#2.4 `LabelEncoder` is used to convert the dignosis types B(Benign) and M(Malignant) to 0 and 1.

le = LabelEncoder()
cancer["diagnosis"] = le.fit_transform(cancer["diagnosis"])
cancer.head()


# In[14]:


#2.5 `cancer.duplicated().sum()` is used to check if there are any duplicate values.

if cancer.duplicated().sum() > 0:
    print("Duplicates found!")
else:
    print("No duplicates found.")


# ### 3. Feature selection

# In[15]:


#3.1 'sns.boxplot' function from the seaborn library, where the x-axis represents the diagnosis column, the y-axis represents the current column being looped over.

fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 15))

columns1 = cancer.columns[1:] # Exclude first column (diagnosis)

for i, column in enumerate(columns1):
    row = i // 6
    col = i % 6
    ax = axes[row, col]
    sns.boxplot(x='diagnosis', y=column, data=cancer, palette='GnBu', hue='diagnosis', ax=ax)
    ax.set_title(column)

plt.tight_layout()
plt.show()


# In[17]:

#3.2A 'stats.zscore' is used to fetch zscore, and plotted using 'plt.subplots'.

# Calculate the z-scores
z = stats.zscore(cancer)
z_scores = pd.DataFrame(z, columns = cancer.columns)

# Plot the z-scores using a box plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=z_scores, orient="h", palette="Set2", ax=ax)
ax.set_title("Z-Scores Box Plot")
plt.show()


# In[18]:

#3.2B Using np.where to create a index using threshold.

threshold = 4
outliers_index = np.where((z < -threshold) | (z > threshold))
outliers_count = len(outliers_index[0])
print("Number of outliers(B): ", outliers_count)


# In[19]:

#3.3 calculates the lower and upper limits for each column in dataset using the interquartile range (IQR) method.

cancer_data = cancer.drop(columns=['diagnosis'])

lower_lim = {}
upper_lim = {}
for col in cancer_data:
    q1 = cancer_data[col].quantile(0.25)
    q3 = cancer_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_lim[col] = max(q1 - 2 * iqr, cancer_data[col].min())
    upper_lim[col] = min(q3 + 2 * iqr, cancer_data[col].max())

lower_lim_df = pd.DataFrame(list(lower_lim.items()),columns=['Column','Lower Limit'])
upper_lim_df = pd.DataFrame(list(upper_lim.items()),columns=['Column','Upper Limit'])

# concatenate the two dataframe
result = pd.concat([lower_lim_df, upper_lim_df], axis=1)
#convert the dataframe to HTML table format and display it in the notebook
display(HTML(result.to_html()))


# In[20]:

#3.4 remove outliers from the new dataset, based on the lower and upper limits calculated in the previous code block, along with droping any null value created. Also assign the 'diagnosis' column back to the new dataset.

cancer_data = cancer_data[(cancer_data > lower_lim) & (cancer_data < upper_lim)].dropna()
cancer_data = cancer_data.assign(diagnosis = cancer['diagnosis']).dropna()
cancer_data

# In[23]:

#3.5A Again using stats.zscore, calculate outliers of new dataset.

z1 = stats.zscore(cancer_data)
z_scores1 = pd.DataFrame(z1, columns = cancer_data.columns)

threshold = 4

outliers_index = np.where((z1 < -threshold) | (z1 > threshold))
outliers_count = len(outliers_index[0])
print("Number of outliers(B): ", outliers_count)


# In[24]:

#3.5B Using 'plt.subplots' to visualize the new zscore.

# Plot the z-scores using a box plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=z_scores1, orient="h", palette="Set2", ax=ax)
ax.set_title("Z-Scores Box Plot")
plt.show()

X = cancer_data.drop(columns='diagnosis')
y = cancer_data['diagnosis']


# In[27]:


#4.2 Split the data into training and testing sets using `train_test_split`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:

#4.3 visualizing the class distribution of the target variable using value_counts and plotting it using plt.bar

# Count the number of instances in each class
class_counts = y_train.value_counts()

# Create a bar plot to visualize the class distribution
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('diagnosis')
plt.ylabel('Number of instances')
plt.show()

# Identify the minority class
minority_class = class_counts.idxmin()
print(f"Minority class: {minority_class}")


# In[29]:

#4.4A The issue of class imbalance is fixed with sm.fit resample 

# using SMOTE Techniqe
sm = SMOTE(random_state=10)
X_sm_train , y_sm_train = sm.fit_resample(X_train,y_train)

print(f'''Target Class distributuion before SMOTE:\n{y_train.value_counts(normalize=True)}
Target Class distributuion after SMOTE :\n{y_sm_train.value_counts(normalize=True)}''')


# In[30]:

#4.4B visualizing the class distribution of the target variable after SMOTE using value_counts and plotting it using plt.bar

# Count the number of instances in each class
class_counts2 = y_sm_train.value_counts()

# Create a bar plot to visualize the class distribution
plt.bar(class_counts2.index, class_counts2.values)
plt.xlabel('diagnosis')
plt.ylabel('Number of instances')
plt.show()


# In[31]:

#4.5 Scaling the data using `StandardScaler()`

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_sm_train).transform(X_sm_train)
X_test_scaled = scaler.transform(X_test)


# In[32]:

#4.6 visualizing the amount of variance captured by each principal component using explained_variance_

pca_check = PCA()
pca_check.fit(X_train_scaled)
pca_components = range(pca_check.n_components_)
plt.bar(pca_components, pca_check.explained_variance_)
plt.xlabel('dimensions')
plt.ylabel('variance')
plt.show()


# In[33]:

#4.6B variance explained by each principal component, along with the corresponding dimension number using the sum of explained_variance_

variance_sum = sum(pca_check.explained_variance_)
combined_variance = 0
for i, component in enumerate(pca_check.explained_variance_):
    combined_variance += (component/variance_sum)*100
    print(f'{round(combined_variance)}% variance is explained by {i+1} dimensions')


# In[34]:

#4.7 performing Principal Component Analysis (PCA) on the standardized training and testing data

pca = PCA(n_components=10)

# Fit and transform the training data
X_train_pca = pca.fit_transform(X_train_scaled)

# Transform the testing data
X_test_pca = pca.transform(X_test_scaled)


# In[35]:

# we use the read function of configParser to load the config file. 

# Load config file
config = ConfigParser()
config.read('Model_smote.ini')


# In[36]:

#4.8 Traning Logistic regression model with hyer parameter.

log_solver = config.get('Logistic', 'solver')
log_penalty = config.get('Logistic', 'penalty')
log_c = [float(c) for c in config.get('Logistic', 'c').split(',')]
log_cv = config.getint('Logistic', 'cv')
log_iter = config.getint('Logistic', 'n_iter')

# Create a logistic regression model
log_reg = LogisticRegression(random_state=20)

# Define the hyperparameter grid to search over
param_dist = {'C': log_c, 'penalty': [log_penalty], 'solver':[log_solver]}

# Create a Randomizedsearch object with cross-validation
random_search = RandomizedSearchCV(log_reg, param_distributions=param_dist, n_iter=log_iter, cv=log_cv,error_score=0)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train_pca, y_sm_train)

# Make predictions on the testing data using the best estimator found by RandomizedSearchCV
log_pred1a = random_search.best_estimator_.predict(X_test_pca)


# best hyperparameters was printed out using below code:
# print("Best parameter combination:", random_search.best_params_)

# In[37]:

#4.9 Traning Random forest model with hyerparameter.

R_estimator = [config.getint('Random', 'n_estimators')]
R_Mfeatures = [config.get('Random', 'max_features')]
R_depth = [config.getint('Random', 'max_depth')]
R_cv = config.getint('Random', 'cv')

rfc = RandomForestClassifier(random_state=20) # Create a RandomForest Classifier model
rfc_params = { # Define the hyperparameter grid to search over
    'n_estimators': R_estimator,
    'max_depth': R_depth,
    'max_features': R_Mfeatures,
} 
grid_search_rfc = GridSearchCV(estimator=rfc,
                               param_grid=rfc_params,
                               cv=R_cv,
                               error_score=0,n_jobs=-1) # Fit the GridSearchCV object to the training data

#performs grid search to find the best hyperparameters for the random forest classifier
grid_search_rfc.fit(X_train_pca, y_sm_train) 
# Make predictions on the testing data using the best estimator found by GridSearchCV
y_pred_rfc = grid_search_rfc.best_estimator_.predict(X_test_pca) 

# best hyperparameters was printed out using below code:
# print(grid_search_rfc.best_params_)

# In[38]:

#4.10 Traning ANN model with hyerparameter.
Ep_deep = config.getint('Deep','epochs')
batc_deep = config.getint('Deep', 'batch_size')
opt_deep = [config.get('Deep','optimizer')]
drop_deep = [float(dropout_rate) for dropout_rate in config.get('Deep', 'dropout_rate').split(',')]
act_deep = [config.get('Deep','activation')]
iter_deep = config.getint('Deep','n_iter')
cv_deep = config.getint('Deep','cv')

# Define the Keras model
def create_ann_model(optimizer='adam', dropout_rate=0.0, activation='relu',random_state=42):
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    model = Sequential()
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define the ANN model as a KerasClassifier object
model = KerasClassifier(build_fn=create_ann_model, epochs=Ep_deep, batch_size=batc_deep, verbose=0,random_state=42)

# Define the hyperparameter search space
param_dist2 = {'optimizer': opt_deep,
              'dropout_rate': drop_deep,
              'activation': act_deep}

# Define the randomized search object
search = RandomizedSearchCV(estimator=model, param_distributions=param_dist2, n_iter=iter_deep, cv=cv_deep, verbose=0, error_score='raise')

# Fit the randomized search object to the data
search.fit(X_train_pca, y_sm_train)

# Predict classes for the test 
y_predBB = search.best_estimator_.predict(X_test_pca)

# best hyperparameters was printed out using below code:
# print("Best Hyperparameters: ", search.best_params_)



# In[39]:

### OPTION B: cross-validation method with pipeline

#5.1 using pd.read_csv and split them into features and target.

X1 = cancer_data.drop(columns='diagnosis')
y1 = cancer_data['diagnosis']


# In[40]:

#5.2 Creating a LogisticRegression model using pipeline 

pipeline_L = Pipeline([
    ('smote', SMOTE(random_state=10)),  # apply SMOTE
    ('scaler', StandardScaler()), # apply scaling
    ('PCA', PCA(n_components=10)),  # apply PCA
    ('clf', LogisticRegression(random_state=20))  # apply logistic regression
])

Logic_acc = cross_validate(pipeline_L, X1, y1, cv=5, scoring='accuracy',return_train_score=True)

# In[41]:

#5.3 Creating a RandomForestClassifier model using pipeline 

# Create a Randomforest model
pipeline_R = Pipeline([
    ('smote', SMOTE(random_state=10)),# apply SMOTE
    ('scaler', StandardScaler()), # apply scaling
    ('PCA', PCA(n_components=10)), # apply PCA 
    ('clf', RandomForestClassifier(random_state=10, n_estimators=50))  # apply Randomclassifier with limiting n_estimtor to aviod overfitting 
])
# Use cross_val_score to perform 5-fold cross-validation and get the scores
Random_acc = cross_validate(pipeline_R, X1, y1, cv=5, scoring='accuracy',return_train_score=True)

# In[42]:

#5.4 Creating a ANN model using pipeline 

def create_ann_model(random_state=20):
    np.random.seed(random_state)
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# create the model
model_deep = KerasClassifier(build_fn=create_ann_model, epochs=10, batch_size=32, verbose=0,random_state=20)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# evaluate the model using 5-fold cross-validation with random state
pipeline_D = Pipeline([
    ('smote', SMOTE(random_state=10)), # apply SMOTE
    ('scaler', StandardScaler()), # apply scaling
    ('PCA', PCA(n_components=10)), # apply PCA,
    ('model', model_deep) # apply ANN
])
# Use cross_val_score to perform 5-fold cross-validation and get the scores
Deep_acc = cross_validate(pipeline_D, X1, y1, cv=5, scoring='accuracy',return_train_score=True)

# In[43]:

### 6. MODELS EVALUATION

#6.1 classification report for testing data: Method 1 

# Generate classification reports for three models
report1a = classification_report(y_test, log_pred1a, output_dict=True, digits=4, zero_division=0)
report2a = classification_report(y_test, y_pred_rfc, output_dict=True, digits=4, zero_division=0)
report3a = classification_report(y_test, y_predBB, output_dict=True, digits=4, zero_division=0)

# Extract precision, recall, f1-score, and accuracy for class 0 and class 1 from the reports
model1_scores1 = [report1a['macro avg']['precision'], report1a['macro avg']['recall'], report1a['macro avg']['f1-score'], report1a['accuracy']]
model2_scores1 = [report2a['macro avg']['precision'], report2a['macro avg']['recall'], report2a['macro avg']['f1-score'], report2a['accuracy']]
model3_scores1 = [report3a['macro avg']['precision'], report3a['macro avg']['recall'], report3a['macro avg']['f1-score'], report3a['accuracy']]

# Create a list of model names and scores
model_names1 = ['Logistic Regression', 'Random Forest', 'ANN']
model_scores1 = [model1_scores1, model2_scores1, model3_scores1]

# Create a list of column headers for the HTML table
headers1 = ['Model', 'Precision ', 'Recall ', 'F1-score', 'Accuracy (Overall)']

# Use the tabulate library to create an HTML table
table1 = tabulate(model_scores1, headers=headers1, showindex=model_names1, tablefmt='html')
banner1 = '<h1>Model Performance : Method 1 </h1>'
table_with_banner = banner1 + table1
# Display the HTML table in a Jupyter Notebook
display(HTML(table_with_banner))


# In[50]:

#6.2 classification report for testing data: Method 2

L_pred = cross_val_predict(pipeline_L, X1, y1, cv=5)
R_pred = cross_val_predict(pipeline_R, X1, y1, cv=5)
D_pred = cross_val_predict(pipeline_D, X1, y1, cv=5)

report1av = classification_report(y1, L_pred, output_dict=True, digits=4, zero_division=0)
report2av = classification_report(y1, R_pred, output_dict=True, digits=4, zero_division=0)
report3av = classification_report(y1, D_pred, output_dict=True, digits=4, zero_division=0)

# Extract precision, recall, f1-score, and accuracy for class 0 and class 1 from the reports
model1_scores1v = [report1av['macro avg']['precision'], report1av['macro avg']['recall'], report1av['macro avg']['f1-score'], report1av['accuracy']]
model2_scores1v = [report2av['macro avg']['precision'], report2av['macro avg']['recall'], report2av['macro avg']['f1-score'], report2av['accuracy']]
model3_scores1v = [report3av['macro avg']['precision'], report3av['macro avg']['recall'], report3av['macro avg']['f1-score'], report3av['accuracy']]

# Create a list of model names and scores
model_names1v = ['Logistic Regression', 'Random Forest', 'ANN']
model_scores1v = [model1_scores1v, model2_scores1v, model3_scores1v]

# Create a list of column headers for the HTML table
headers1v = ['Model', 'Precision ', 'Recall ', 'F1-score', 'Accuracy (Overall)']

# Use the tabulate library to create an HTML table
table1v = tabulate(model_scores1v, headers=headers1v, showindex=model_names1v, tablefmt='html')
banner1v = '<h1>Model Performance: Method 2 </h1>'
table_with_bannerv = banner1v + table1v
# Display the HTML table in a Jupyter Notebook
display(HTML(table_with_bannerv))


# In[50]:

#6.3 Plotting confusion matrix for method 1

# Compute the confusion matrices: Method 1
cmLa = confusion_matrix(y_test, log_pred1a)
cmRa = confusion_matrix(y_test, y_pred_rfc)
#Define the y_test_classes variable as a binary vector to compute the confusion matrix correctly for ANN
y_test_classes = label_binarize(y_test, classes=[0, 1])[:, 0]
cmDa = confusion_matrix(y_test, y_predBB)

# Create a figure with three subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot the first confusion matrix as heatmap
sns.heatmap(cmLa, annot=True, cmap='Blues', fmt='g', ax=axes[0])
axes[0].set_xlabel('Predicted labels')
axes[0].set_ylabel('True labels')
axes[0].set_title('Confusion matrix for Logstic Regression')

# Plot the second confusion matrix as heatmap
sns.heatmap(cmRa, annot=True, cmap='Blues', fmt='g', ax=axes[1])
axes[1].set_xlabel('Predicted labels')
axes[1].set_ylabel('True labels')
axes[1].set_title('Confusion matrix for Randomforest')

# Plot the third confusion matrix as heatmap
sns.heatmap(cmDa, annot=True, cmap='Blues', fmt='g', ax=axes[2])
axes[2].set_xlabel('Predicted labels')
axes[2].set_ylabel('True labels')
axes[2].set_title('Confusion matrix for ANN')

# Show the plot
plt.show()

# In[51]:

#6.4 Plotting confusion matrix for method 2

# Compute the confusion matrices

cm_L = confusion_matrix(y1, L_pred)
cm_R = confusion_matrix(y1, R_pred)
cm_D = confusion_matrix(y1, D_pred)


# Create a figure with three subplots : OPTION B
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot the first confusion matrix as heatmap
sns.heatmap(cm_L, annot=True, cmap='Blues', fmt='g', ax=axes[0])
axes[0].set_xlabel('Predicted labels')
axes[0].set_ylabel('True labels')
axes[0].set_title('Confusion matrix for Logstic Regression')

# Plot the second confusion matrix as heatmap
sns.heatmap(cm_R, annot=True, cmap='Blues', fmt='g', ax=axes[1])
axes[1].set_xlabel('Predicted labels')
axes[1].set_ylabel('True labels')
axes[1].set_title('Confusion matrix for Randomforest')

# Plot the third confusion matrix as heatmap
sns.heatmap(cm_D, annot=True, cmap='Blues', fmt='g', ax=axes[2])
axes[2].set_xlabel('Predicted labels')
axes[2].set_ylabel('True labels')
axes[2].set_title('Confusion matrix for ANN')

# Show the plot
plt.show()

# %%

#6.5 plotting the true positive rate against the false positive rate using ROC-AUC

fpr1, tpr1, thresholds1 = roc_curve(y_test, log_pred1a)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_rfc)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_predBB)

roc_auc1 = roc_auc_score(y_test, log_pred1a)
roc_auc2 = roc_auc_score(y_test, y_pred_rfc)
roc_auc3 = roc_auc_score(y_test, y_predBB)

# Plot the ROC curves for each model
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='Logistic (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue', lw=2, label='Random (AUC = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='green', lw=2, label='KNN (AUC = %0.2f)' % roc_auc3)

# Add the diagonal line representing a random classifier
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set the limits and labels of the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# Add a legend to the plot
plt.legend(loc="lower right")

# Show the plot
plt.show()


# In[45]:

#6.6A cross-validation score for method 1

new = random_search.best_estimator_
new2 = grid_search_rfc.best_estimator_
new3  = search.best_estimator_

models = [new3, new2, new]
model_names = ['Deep Learning', 'Random', 'Logistic Regression']

# Initialize the figure
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot the boxplots
for i, model in enumerate(models):
    # Get the cross-validation scores
    scores = cross_val_score(model, X_train_pca, y_sm_train, cv=4, scoring='accuracy')
    # Plot the boxplot
    axs[i].boxplot(scores)
    axs[i].set_title(model_names[i])
    axs[i].set_ylabel('Score')

# Show the plots
plt.tight_layout()
plt.show()

# In[52
# %%

#6.6B cross-validation score for method 1

# Compute the cross-validation scores Method 1
scores1 = cross_val_score(new, X_train_pca, y_sm_train, cv=4)
scores2 = cross_val_score(new2, X_train_pca, y_sm_train, cv=4)
scores3 = cross_val_score(new3, X_train_pca, y_sm_train, cv=4)

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the cross-validation scores as line plots
ax.plot(scores1, label='Logstic Regression')
ax.plot(scores2, label='Random')
ax.plot(scores3, label='ANN')

# Set the axis labels and title
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Scores')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# In[47]:

#6.7A cross-validation score for method 2

# Define the models
modelsb = [pipeline_D, pipeline_R, pipeline_L]
model_namesb = ['Deep Learning', 'Random', 'Logistic Regression']

# Initialize the figure
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot the boxplots
for i, modelb in enumerate(modelsb):
    # Get the cross-validation scores
    scoresb = cross_val_score(modelb, X1, y1, cv=4, scoring='accuracy')
    # Plot the boxplot
    axs[i].boxplot(scoresb)
    axs[i].set_title(model_namesb[i])
    axs[i].set_ylabel('Score')

# Show the plots
plt.tight_layout()
plt.show()


# %%

#6.7B cross-validation score for method 2

# Compute the cross-validation scores
scores1b = cross_val_score(pipeline_L, X1, y1, cv=4)
scores2b = cross_val_score(pipeline_R, X1, y1, cv=4)
scores3b = cross_val_score(pipeline_D, X1, y1, cv=4)

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the cross-validation scores as line plots
ax.plot(scores1b, label='Logstic Regression')
ax.plot(scores2b, label='Random')
ax.plot(scores3b, label='ANN')

# Set the axis labels and title
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Scores')

# Add a legend
ax.legend()

# Show the plot
plt.show()