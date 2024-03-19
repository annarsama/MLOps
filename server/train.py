import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data in 3 different dataframes
adelie = pd.read_csv('/Users/annabellenarsama/Desktop/MLOps//data/table_219.csv')
gentoo = pd.read_csv('/Users/annabellenarsama/Desktop/MLOps//data/table_220.csv')
chinstrap = pd.read_csv('/Users/annabellenarsama/Desktop/MLOps/data/table_221.csv')

# Shape of the 3 datasets
adelie.shape
gentoo.shape
chinstrap.shape

# Merge the 3 dataframes
df = pd.concat([adelie,gentoo,chinstrap], axis=0, ignore_index=True)
df.head()

# Shape of the merged df
df.shape

# Drop unuseful features
data = df.drop(columns=['studyName', 'Sample Number', 'Region', 'Island', 'Stage',
       'Individual ID', 'Clutch Completion', 'Date Egg', 'Sex', 'Comments', 'Delta 13 C (o/oo)'])
data.head()

# Descriptive statistics
data.describe()

# Min and max values for each feature
print('Max :', data['Culmen Depth (mm)'].max())
print('Min :', data['Culmen Depth (mm)'].min())

print('Max :', data['Culmen Length (mm)'].max())
print('Min :', data['Culmen Length (mm)'].min())

print('Max :', data['Flipper Length (mm)'].max())
print('Min :', data['Flipper Length (mm)'].min())

print('Max :', data['Body Mass (g)'].max())
print('Min :', data['Body Mass (g)'].min())

print('Max :', data['Delta 15 N (o/oo)'].max())
print('Min :', data['Delta 15 N (o/oo)'].min())

# Shape of the final dataframe
data.shape

# Features-label split
X = data.select_dtypes(exclude="object")
y = data.select_dtypes(exclude="float")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    stratify = y,
                                                    random_state = 0)

# Remplacement des NaN

# Échantillon d'apprentissage :
for col in X_train :
    X_train[col] = X_train[col].fillna(X_train[col].mean())

# Vérification du remplacement des NaN :
print(X_train.isna().sum()) # 0 NaN

# Échantillon test :
for col in X_test :
    X_test[col] = X_test[col].fillna(X_test[col].mean())

# Vérification du remplacement des NaN :
print("\n", X_test.isna().sum()) # 0 NaN

# Info about the features
X_train.info()

# Info about the labels
y_train.info()

# Train ML model
clf = LogisticRegression(random_state=0)
model = clf.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')