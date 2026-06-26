import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/empath.csv", encoding = "ISO-8859-1")

# Clean up index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove rows without valid status text
df.dropna(subset=['STATUS'], inplace=True)

# List all user-level columns that are constant per user
userLevelColumns = [
    'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN', 
    'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN',
    'NETWORKSIZE', 'BETWEENNESS', 'DENSITY', 'BROKERAGE', 'TRANSITIVITY'
]
userLevelColumns = [col for col in userLevelColumns if col in df.columns]

# Build aggregation rules dynamically
aggRules = {}
for col in df.columns:
    if col == '#AUTHID' or col == 'STATUS' or col == 'DATE':
        continue
    elif col in userLevelColumns:
        aggRules[col] = 'first'
    elif pd.api.types.is_numeric_dtype(df[col]):
        aggRules[col] = 'mean'

# Group by user ID (#AUTHID) to create a user-level dataset
groupedDf = df.groupby('#AUTHID').agg(aggRules).reset_index()

# Create plots folder
os.makedirs("plots", exist_ok = True)

# Set global plot styling for high-quality visuals
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# VISUALIZATION 1: HISTOGRAM (Distribution of User Average Status Word Counts)
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.hist(groupedDf['STATUS_WORD_LEN'], bins=25, color='#4A90E2', edgecolor='white', alpha=0.85, rwidth=0.9)
plt.axvline(groupedDf['STATUS_WORD_LEN'].mean(), color='#D0021B', linestyle='--', linewidth=1.5, label=f"Mean: {groupedDf['STATUS_WORD_LEN'].mean():.2f} words")
plt.title("Distribution of User Average Status Word Length", fontsize=14, fontweight='bold', pad=15, color='#333333')
plt.xlabel("Average Word Count per Status Update", fontsize=11, labelpad=10)
plt.ylabel("Number of Users", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("plots/1_status_word_len_distribution.png", dpi=150)
plt.close()

# ==============================================================================
# VISUALIZATION 2: SCATTER PLOT WITH TREND LINE (Word Count vs. Openness Score)
# ==============================================================================
plt.figure(figsize=(8, 6))
xVals = groupedDf['sOPN']
yVals = groupedDf['STATUS_WORD_LEN']
plt.scatter(xVals, yVals, color='#7ED321', alpha=0.7, edgecolors='none', s=60, label="User profile")
slope, intercept = np.polyfit(xVals, yVals, 1)
trendline = np.poly1d([slope, intercept])
xRange = np.linspace(xVals.min(), xVals.max(), 100)
plt.plot(xRange, trendline(xRange), color='#4A90E2', linestyle='-', linewidth=2, label=f"Trend (slope: {slope:.2f})")
plt.title("Status Verbosity vs. Openness to Experience (sOPN)", fontsize=14, fontweight='bold', pad=15, color='#333333')
plt.xlabel("Openness Score (1.0 to 5.0)", fontsize=11, labelpad=10)
plt.ylabel("Average Status Word Count", fontsize=11, labelpad=10)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("plots/2_verbosity_vs_openness.png", dpi=150)
plt.close()

# ==============================================================================
# VISUALIZATION 3: BOX PLOT (Network Size by Extraversion Classification)
# ==============================================================================
plt.figure(figsize=(7, 6))
netSizeExtN = groupedDf[groupedDf['cEXT'] == 'n']['NETWORKSIZE'].dropna()
netSizeExtY = groupedDf[groupedDf['cEXT'] == 'y']['NETWORKSIZE'].dropna()
boxData = [netSizeExtN, netSizeExtY]
box = plt.boxplot(boxData, tick_labels=["Introverted (cEXT='n')", "Extraverted (cEXT='y')"], 
                  patch_artist=True, medianprops={'color': '#D0021B', 'linewidth': 1.5},
                  boxprops={'color': '#4A90E2', 'linewidth': 1.2},
                  whiskerprops={'color': '#999999', 'linestyle': '--'})
colors = ['#E5F1FD', '#E2F7E3']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.title("Social Network Size: Introverts vs. Extraverts", fontsize=14, fontweight='bold', pad=15, color='#333333')
plt.ylabel("Network Size (Number of Friends)", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/3_network_size_by_extraversion.png", dpi=150)
plt.close()

# ==============================================================================
# VISUALIZATION 4: GROUPED BAR CHART (Linguistic Markers vs. Neuroticism Group)
# ==============================================================================
plt.figure(figsize=(8, 6))
neuGroups = groupedDf.groupby('cNEU').mean(numeric_only=True)
categories = ['Introvert Pronouns', 'Positive Polarity', 'Subjectivity']
xIndices = np.arange(2)
barWidth = 0.25
pronounsY = groupedDf[groupedDf['cNEU'] == 'y']['FIRST_PERSON_PRONOUN_DENSITY'].mean()
pronounsN = groupedDf[groupedDf['cNEU'] == 'n']['FIRST_PERSON_PRONOUN_DENSITY'].mean()
exclY = groupedDf[groupedDf['cNEU'] == 'y']['EXCLAMATION_COUNT'].mean()
exclN = groupedDf[groupedDf['cNEU'] == 'n']['EXCLAMATION_COUNT'].mean()
plt.bar(xIndices - barWidth/2, [pronounsN, pronounsY], width=barWidth, color='#F5A623', label='1st Person Pronoun Density', alpha=0.85)
plt.bar(xIndices + barWidth/2, [exclN, exclY], width=barWidth, color='#D0021B', label='Average Exclamation Marks', alpha=0.85)
plt.title("Linguistic & Punctuation Style by Neuroticism Class (cNEU)", fontsize=14, fontweight='bold', pad=15, color='#333333')
plt.xticks(xIndices, ["Low Neuroticism (cNEU='n')", "High Neuroticism (cNEU='y')"], fontsize=11)
plt.ylabel("Average Value of Features", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("plots/4_linguistics_by_neuroticism.png", dpi=150)
plt.close()

# ==============================================================================
# VISUALIZATION 5: CORRELATION HEATMAP
# ==============================================================================
selectedFeatures = [
    'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN',
    'NETWORKSIZE', 'BETWEENNESS', 'DENSITY',
    'AVG_WORD_LEN', 'EXCLAMATION_COUNT', 'FIRST_PERSON_PRONOUN_DENSITY',
    'aggression', 'swearing_terms', 'politeness', 'emotional'
]
selectedFeatures = [col for col in selectedFeatures if col in groupedDf.columns]
corrMatrix = groupedDf[selectedFeatures].corr()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(corrMatrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')
for i in range(len(selectedFeatures)):
    for j in range(len(selectedFeatures)):
        ax.text(j, i, f"{corrMatrix.iloc[i, j]:.2f}", ha='center', va='center', 
                color='black' if abs(corrMatrix.iloc[i, j]) < 0.5 else 'white', fontsize=9)
colorbar = fig.colorbar(cax, shrink=0.8)
colorbar.set_label("Pearson Correlation Coefficient", fontsize=11, labelpad=10)
ax.set_xticks(np.arange(len(selectedFeatures)))
ax.set_yticks(np.arange(len(selectedFeatures)))
ax.set_xticklabels(selectedFeatures, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(selectedFeatures, fontsize=9)
plt.title("Correlation Matrix of Personality, Network, and Writing Style", fontsize=14, fontweight='bold', pad=20, color='#333333')
plt.tight_layout()
plt.savefig("plots/5_feature_correlation_heatmap.png", dpi=150)
plt.close()