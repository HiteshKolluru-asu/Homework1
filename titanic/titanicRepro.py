import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv')
print("Data shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Deck'] = df['Cabin'].astype(str).str[0]
df['Deck'] = df['Deck'].replace('n', 'U').fillna('U')

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

df['Age'] = cap_outliers(df['Age'])
df['Fare'] = cap_outliers(df['Fare'])

df['Fare_log'] = np.log1p(df['Fare'])

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
)
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df['Pclass_Age'] = df['Pclass'] * df['Age']

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked', 'Deck', 'Title'], drop_first=True)

scaler = StandardScaler()
df['Fare_log_scaled'] = scaler.fit_transform(df[['Fare_log']])

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', 'Fare_log'], inplace=True)

print("Processed data sample:")
print(df.head())
print("Processed data info:")
print(df.info())
print("Missing values after processing:\n", df.isnull().sum())

X = df.drop('Survived', axis=1)
y = df['Survived']

dtree = DecisionTreeClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtree.fit(X_train, y_train)
print("Decision Tree Train Accuracy:", dtree.score(X_train, y_train))
print("Decision Tree Test Accuracy:", dtree.score(X_test, y_test))

cv_scores = cross_val_score(dtree, X_train, y_train, cv=5)
print("Decision Tree 5-fold CV Scores:", cv_scores)
print("Decision Tree Average CV Score:", round(cv_scores.mean() * 100, 2), "%")

plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=['Not Survived','Survived'], filled=True)
plt.show()

rf = RandomForestClassifier(random_state=42)
rf_cv_scores = cross_val_score(rf, X, y, cv=5)
print("Random Forest 5-fold CV Scores:", rf_cv_scores)
print("Random Forest Average CV Score:", round(rf_cv_scores.mean() * 100, 2), "%")
