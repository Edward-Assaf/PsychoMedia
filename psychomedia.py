import pandas as pd
from empath import Empath
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Setting debugging configurations for Pandas
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Loading raw dataset
print("Loading dataset...")
dataset = pd.read_csv("datasets/fbDataset.csv", encoding="ISO-8859-1")

# Fix a shifted row numbered 6399
print("Performing cleaning...")
for i in range(len(dataset.iloc[6399]) - 2, 12, -1):
    dataset.iloc[6399, i + 1] = dataset.iloc[6399, i]
dataset.iloc[6399, 13] = None

# Removing unnecessary column 'Unnamed'
dataset.drop("Unnamed: 0", axis = 1, inplace = True)

# Separating the DATE feature into multiple features (YEAR, MONTH, DAY, HOUR, MINUTE)
dataset['DATE'] = pd.to_datetime(dataset['DATE'])
dataset['YEAR'] = dataset['DATE'].dt.year
dataset['MONTH'] = dataset['DATE'].dt.month
dataset['DAY'] = dataset['DATE'].dt.day
dataset['HOUR'] = dataset['DATE'].dt.hour
dataset['MINUTE'] = dataset['DATE'].dt.minute

# Deleting rows with empty STATUS feature
dataset.dropna(subset=['STATUS'], inplace = True)

# Filling missing DATE values in DATE, for every missing DATE value for every user
dateFeatures = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']
groupFeatures = ['#AUTHID']
for dateFeature in dateFeatures:
    mostFrequentLookup = dataset.groupby(groupFeatures)[dateFeature].agg(
        lambda x : x.mode().iloc[0] if not x.mode().empty else None
    )
    usersMostFrequent = dataset[groupFeatures].join(mostFrequentLookup, on = groupFeatures)[dateFeature]
    dataset[dateFeature] = dataset[dateFeature].fillna(usersMostFrequent)
    groupFeatures.append(dateFeature)

# Filling missing numerical columns, as an average per user
numericColumns = ['BETWEENNESS', 'DENSITY']
userMeans = dataset.groupby("#AUTHID")[numericColumns].transform('mean')
dataset[numericColumns] = dataset[numericColumns].fillna(userMeans)
globalMeans = dataset[numericColumns].mean()
dataset[numericColumns] = dataset[numericColumns].fillna(globalMeans)

# Use empath to extract psychological metrics from the text stored in STATUS
print("Extracting metrics from STATUS using Empath...")
lexicon = Empath()
categories = list(lexicon.cats.keys())

def safeAnalyze(text):
    if not isinstance(text, str) or not text.strip():
        return {cat: 0.0 for cat in categories}
    analyzed = lexicon.analyze(text, normalize=True)
    if analyzed is None:
        return {cat: 0.0 for cat in categories}
    return analyzed

empathSeries = dataset['STATUS'].apply(safeAnalyze)
empathFeatures = pd.DataFrame(empathSeries.tolist(), index=dataset.index)
dataset = pd.concat([dataset, empathFeatures], axis=1)

# Feature-engineering STATUS length metrics
print("Feature-engineering new columns...")
dataset['STATUS_CHAR_LEN'] = dataset['STATUS'].str.len()
dataset['STATUS_WORD_LEN'] = dataset['STATUS'].apply(lambda x: len(str(x).split()))
dataset['AVG_WORD_LEN'] = dataset['STATUS_CHAR_LEN'] / dataset['STATUS_WORD_LEN'].replace(0, 1)

# Feature-engineering typing-style metrics
dataset['EXCLAMATION_COUNT'] = dataset['STATUS'].apply(lambda x: str(x).count('!'))
dataset['QUESTION_COUNT'] = dataset['STATUS'].apply(lambda x: str(x).count('?'))

def getUppercaseRatio(text):
    textString = str(text)
    totalChars = len(textString)
    if totalChars == 0:
        return 0.0
    upperChars = sum(1 for c in textString if c.isupper())
    return upperChars / totalChars

dataset['UPPERCASE_RATIO'] = dataset['STATUS'].apply(getUppercaseRatio)

def getFirstPersonPronounsDensity(text, wordLen):
    if wordLen <= 0 or not isinstance(text, str):
        return 0.0
    # Case-insensitive match for word boundaries around pronouns
    pronouns = r'\b(i|me|my|mine|myself)\b'
    words = re.findall(pronouns, text.lower())
    return len(words) / wordLen

dataset['FIRST_PERSON_PRONOUN_DENSITY'] = dataset.apply(
    lambda row: getFirstPersonPronounsDensity(row['STATUS'], row['STATUS_WORD_LEN']), axis=1
)

# Feature-engineering weekdays-specific metrics based on the DATE feature
dataset['DAY_OF_WEEK'] = dataset['DATE'].dt.dayofweek
dataset['IS_WEEKEND'] = dataset['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)
dataset['IS_NIGHT'] = dataset['HOUR'].apply(lambda h: 1 if (h >= 23 or h < 5) else 0)

# Grouping all data by username
print("Preparing dataset for learning (grouping, splitting, scaling)...")
userLevelColumns = [
    'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 
    'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN',
    'NETWORKSIZE', 'BETWEENNESS', 'DENSITY', 'BROKERAGE', 'TRANSITIVITY'
]
userLevelColumns = [col for col in userLevelColumns if col in dataset.columns]
aggregationRules = {}
for column in dataset.columns:
    if column == '#AUTHID' or column == 'STATUS':
        continue
    elif column in userLevelColumns:
        aggregationRules[column] = 'first'
    elif pd.api.types.is_numeric_dtype(dataset[column]):
        aggregationRules[column] = 'mean'
groupedDataset = dataset.groupby('#AUTHID').agg(aggregationRules).reset_index()

# Splitting dataset into training set and testing set
labels = ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']
features = [
    col for col in groupedDataset.select_dtypes(include=['number']).columns 
    if col not in labels
]
x = groupedDataset[features]
y = groupedDataset[labels]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state = 42)

# Scaling and normalizing numeric columns
columnsToScale = []
for column in xTrain.columns:
    minimum = xTrain[column].min()
    maximum = xTrain[column].max()
    if pd.isna(minimum) or pd.isna(maximum):
        continue
    if minimum < -1e-9 or maximum > 1.0 + 1e-9:
        columnsToScale.append(column)
scaler = MinMaxScaler()
if columnsToScale:
    xTrainScaled = xTrain.copy()
    xTestScaled = xTest.copy()
    xTrainScaled[columnsToScale] = scaler.fit_transform(xTrain[columnsToScale])
    xTestScaled[columnsToScale] = scaler.transform(xTest[columnsToScale])
else:
    xTrainScaled = xTrain
    xTestScaled = xTest
xTrainScaled.to_csv("datasets/training.csv")
xTestScaled.to_csv("datasets/testing.csv")

# Initializing and training models
print("Training RidgeRegression and RandomForestRegressor models...")
ridge = Ridge()
forest = RandomForestRegressor(random_state=42,n_jobs=-1,n_estimators=300,max_depth=13,min_samples_leaf=2)
ridge.fit(xTrainScaled, yTrain)
forest.fit(xTrainScaled, yTrain)
ridgePredictions = ridge.predict(xTestScaled)
forestPredictions = forest.predict(xTestScaled)
print("Ridge MSE: " + str(mean_squared_error(yTest, ridgePredictions)))
print("Random Forest MSE: " + str(mean_squared_error(yTest, forestPredictions)))

# Saving trained models and preprocessing artifacts
print("Saving models and preprocessing artifacts...")
os.makedirs("models", exist_ok=True)
joblib.dump(ridge, "models/ridge_model.joblib")
joblib.dump(forest, "models/forest_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(features, "models/features.joblib")
joblib.dump(columnsToScale, "models/columns_to_scale.joblib")
print("Models saved to models/ directory.")
print("Done!")