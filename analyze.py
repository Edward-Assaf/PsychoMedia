import pandas as pd
from empath import Empath
import re
import joblib
import random
from datetime import datetime

# Load saved artifacts
print("Loading models and preprocessing artifacts...")
forest = joblib.load("models/forest_model.joblib")
scaler = joblib.load("models/scaler.joblib")
features = joblib.load("models/features.joblib")
columnsToScale = joblib.load("models/columns_to_scale.joblib")
print("Models loaded.\n")

# Helpers (identical to psychomedia.py)
lexicon = Empath()
categories = list(lexicon.cats.keys())

def safeAnalyze(text):
    if not isinstance(text, str) or not text.strip():
        return {cat: 0.0 for cat in categories}
    analyzed = lexicon.analyze(text, normalize=True)
    if analyzed is None:
        return {cat: 0.0 for cat in categories}
    return analyzed

def getUppercaseRatio(text):
    textString = str(text)
    totalChars = len(textString)
    if totalChars == 0:
        return 0.0
    upperChars = sum(1 for c in textString if c.isupper())
    return upperChars / totalChars

def getFirstPersonPronounsDensity(text, wordLen):
    if wordLen <= 0 or not isinstance(text, str):
        return 0.0
    pronouns = r'\b(i|me|my|mine|myself)\b'
    words = re.findall(pronouns, text.lower())
    return len(words) / wordLen

def promptFloat(label, default=None):
    suffix = f" [Default: {default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")

def promptDate(label, default=None):
    suffix = f" [Default: {default}]" if default is not None else ""
    while True:
        raw = input(f"  {label} (YYYY-MM-DD HH:MM){suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return datetime.strptime(raw, "%Y-%m-%d %H:%M")
        except ValueError:
            print("Format must be YYYY-MM-DD HH:MM (e.g. 2024-03-15 14:30).")

while True:
    try:
        n = int(input("How many Facebook posts do you want to enter? ").strip())
        if n < 1:
            raise ValueError
        break
    except ValueError:
        print("Please enter a positive integer.\n")

# Collect user-level network features via intuitive questions
print("\n--- Social Network Questions ---")
print("(These help estimate your position in your social network.)\n")

# NETWORKSIZE: direct count
print("Q1. Approximately how many Facebook friends do you have?")
print("    (Enter the total number, e.g. 250)")
NETWORKSIZE = promptFloat("Friends count", 150)

# BETWEENNESS: how often you bridge others — direct 0.0–1.0
print("\nQ2. How often do you act as a connector between people who don't know")
print("    each other — e.g. introducing friends from different circles?")
print("    0.0 = Never   0.25 = Occasionally   0.5 = Sometimes")
print("    0.75 = Often  1.0  = You are a key bridge in your network")
betweenRaw = None
while betweenRaw is None:
    val = promptFloat("Your answer (0.0–1.0)", 0.5)
    if 0.0 <= val <= 1.0:
        betweenRaw = val
    else:
        print("Please enter a value between 0.0 and 1.0.")
BETWEENNESS = betweenRaw

# DENSITY: how interconnected are your friends — direct 0.0–1.0
print("\nQ3. How well do your friends know each other (across all your circles)?")
print("    0.0 = Circles are completely separate   0.25 = Little overlap")
print("    0.5 = Moderate overlap                  0.75 = Most friends know each other")
print("    1.0 = Almost everyone knows everyone")
densityRaw = None
while densityRaw is None:
    val = promptFloat("Your answer (0.0–1.0)", 0.5)
    if 0.0 <= val <= 1.0:
        densityRaw = val
    else:
        print("Please enter a value between 0.0 and 1.0.")
DENSITY = densityRaw

# BROKERAGE: number of distinct social groups you bridge
print("\nQ4. How many distinct social groups do you feel you bridge or connect?")
print("    (e.g. work colleagues, university friends, sports team, family, etc.)")
print("    Enter a whole number, e.g. 3")
brokerageRaw = None
while brokerageRaw is None:
    val = promptFloat("Number of groups bridged", 2)
    if val >= 0:
        brokerageRaw = val
    else:
        print("Please enter 0 or a positive number.")
BROKERAGE = brokerageRaw

# TRANSITIVITY: clustering coefficient — % of friend-pairs that also know each other
print("\nQ5. Out of all possible pairs among your friends, roughly what percentage")
print("    also know each other? (0 = none, 100 = all)")
transRaw = None
while transRaw is None:
    val = promptFloat("Percentage (0–100)", 30)
    if 0 <= val <= 100:
        transRaw = val
    else:
        print("Please enter a value between 0 and 100.")
TRANSITIVITY = transRaw / 100   # maps [0,100] → [0.0, 1.0]

# Collect per-post rows
nowStr = datetime.now().strftime("%Y-%m-%d %H:%M")
rows = []

for i in range(1, n + 1):
    status = input("STATUS text: ").strip()
    if not status:
        status = " "   # keep a non-empty placeholder
    dt = promptDate("  DATE", default=nowStr)
    if isinstance(dt, str):                     # user hit Enter → parse the default
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M")

    rows.append({
        "STATUS": status,
        "DATE": dt,
        "NETWORKSIZE": NETWORKSIZE,
        "BETWEENNESS": BETWEENNESS,
        "DENSITY": DENSITY,
        "BROKERAGE": BROKERAGE,
        "TRANSITIVITY": TRANSITIVITY,
    })

# Build DataFrame
print("Processing...")
df = pd.DataFrame(rows)
df["DATE"] = pd.to_datetime(df["DATE"])

# Date decomposition
df["YEAR"] = df["DATE"].dt.year
df["MONTH"] = df["DATE"].dt.month
df["DAY"] = df["DATE"].dt.day
df["HOUR"] = df["DATE"].dt.hour
df["MINUTE"] = df["DATE"].dt.minute
df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek
df["IS_WEEKEND"] = df["DAY_OF_WEEK"].apply(lambda x: 1 if x >= 5 else 0)
df["IS_NIGHT"] = df["HOUR"].apply(lambda h: 1 if (h >= 23 or h < 5) else 0)

# Empath features
empathSeries = df["STATUS"].apply(safeAnalyze)
empathFeatures = pd.DataFrame(empathSeries.tolist(), index=df.index)
df = pd.concat([df, empathFeatures], axis=1)

# Status length / style features
df["STATUS_CHAR_LEN"] = df["STATUS"].str.len()
df["STATUS_WORD_LEN"] = df["STATUS"].apply(lambda x: len(str(x).split()))
df["AVG_WORD_LEN"] = df["STATUS_CHAR_LEN"] / df["STATUS_WORD_LEN"].replace(0, 1)
df["EXCLAMATION_COUNT"] = df["STATUS"].apply(lambda x: str(x).count("!"))
df["QUESTION_COUNT"] = df["STATUS"].apply(lambda x: str(x).count("?"))
df["UPPERCASE_RATIO"] = df["STATUS"].apply(getUppercaseRatio)
df["FIRST_PERSON_PRONOUN_DENSITY"] = df.apply(
    lambda row: getFirstPersonPronounsDensity(row["STATUS"], row["STATUS_WORD_LEN"]), axis=1
)

# Group all posts into one user row
userLevelCols = [
    "NETWORKSIZE", "BETWEENNESS", "DENSITY", "BROKERAGE", "TRANSITIVITY",
]
userLevelCols = [c for c in userLevelCols if c in df.columns]

aggRules = {}
for col in df.columns:
    if col in ("STATUS", "DATE"):
        continue
    elif col in userLevelCols:
        aggRules[col] = "first"
    elif pd.api.types.is_numeric_dtype(df[col]):
        aggRules[col] = "mean"

# Add a dummy grouping key so we can use groupby
df["_USER"] = "user"
grouped = df.groupby("_USER").agg(aggRules).reset_index(drop=True)

# Align to the exact feature set the models were trained on
# Add any missing columns as 0, then select in training order
for col in features:
    if col not in grouped.columns:
        grouped[col] = 0.0

xInput = grouped[features]

# Scale
xScaled = xInput.copy()
if columnsToScale:
    colsPresent = [c for c in columnsToScale if c in xScaled.columns]
    if colsPresent:
        xScaled[colsPresent] = scaler.transform(xScaled[colsPresent])

# Predict
labels = ["sEXT", "sNEU", "sAGR", "sCON", "sOPN"]
labelNames = {
    "sEXT": "Extraversion",
    "sNEU": "Neuroticism",
    "sAGR": "Agreeableness",
    "sCON": "Conscientiousness",
    "sOPN": "Openness",
}

forestPred = forest.predict(xScaled)[0]

print("Personality Traits:")
scores = {}
for i, label in enumerate(labels):
    name = labelNames[label]
    # Apply safeguard to keep final outputs in known [1.0, 5.0] OCEAN range
    clippedScore = max(1.0, min(5.0, forestPred[i]))
    scores[label] = clippedScore
    print(f"{name}: {forestPred[i]:.4f} (RandomForest) -> Avg: {clippedScore:.4f}")

# Generate visual card
from personality_card import generatePersonalityCard

print("\n--- Visual Personality Card Generation ---")
userName = input("Enter your name (for the personality card) [Default: Guest User]: ").strip()
if not userName:
    userName = "Guest User"

outputPath = f"outputs/{userName}_personality_report.png"
generatePersonalityCard(userName, random.randint(100001, 1000000), scores, outputPath=outputPath)