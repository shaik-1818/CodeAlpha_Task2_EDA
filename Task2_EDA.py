# =============================================================================
# CodeAlpha Data Analytics Internship
# TASK 2 — Exploratory Data Analysis (EDA)
# Dataset : IPL 2025 Batting Statistics
# Tools   : Python, Pandas, NumPy, Matplotlib, Seaborn
# Author  : [Your Name]
# =============================================================================

# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

os.makedirs('eda_outputs', exist_ok=True)

print("=" * 60)
print("   CodeAlpha — Task 2 : Exploratory Data Analysis")
print("   Dataset : IPL 2025 Batting Statistics")
print("=" * 60)

# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
print("\n[STEP 1] Loading Dataset...")

df = pd.read_csv('IPL2025Batters.csv')

print(f"  ✅ Dataset loaded successfully!")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print()
print(df.head().to_string())

# =============================================================================
# STEP 2 — UNDERSTAND DATA STRUCTURE
# =============================================================================
print("\n" + "=" * 60)
print("[STEP 2] Data Structure & Column Info")
print("=" * 60)

print("\nColumn Names & Data Types:")
print(df.dtypes.to_string())

print("\nColumn Descriptions:")
col_desc = {
    'Player Name' : 'Name of the batter',
    'Team'        : 'IPL franchise (10 teams)',
    'Runs'        : 'Total runs scored in the tournament',
    'Matches'     : 'Number of matches played',
    'Inn'         : 'Number of innings batted',
    'No'          : 'Number of not-out innings',
    'HS'          : 'Highest score (* = not out)',
    'AVG'         : 'Batting average (Runs / dismissals)',
    'BF'          : 'Balls faced',
    'SR'          : 'Strike rate (Runs / BF * 100)',
    '100s'        : 'Number of centuries scored',
    '50s'         : 'Number of fifties scored',
    '4s'          : 'Number of fours hit',
    '6s'          : 'Number of sixes hit',
}
for col, desc in col_desc.items():
    print(f"  {col:<15} : {desc}")

print(f"\nTeams in dataset: {sorted(df['Team'].unique().tolist())}")
print(f"Total Teams     : {df['Team'].nunique()}")

# =============================================================================
# STEP 3 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("[STEP 3] Data Cleaning")
print("=" * 60)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().to_string())

# Duplicates
print(f"\nDuplicate Rows : {df.duplicated().sum()}")

# Fix 1: HS column has '*' for not-out — strip and convert to int
df['HS_numeric'] = df['HS'].str.replace('*', '', regex=False).astype(int)
print("\n✅ HS column cleaned (removed '*' for not-out scores)")

# Fix 2: AVG has '-' for never-dismissed players — convert to NaN
df['AVG_numeric'] = pd.to_numeric(df['AVG'], errors='coerce')
dash_count = df['AVG'].astype(str).str.strip().eq('-').sum()
print(f"✅ AVG column cleaned ({dash_count} players had '-' → converted to NaN)")

print("\nSample cleaned data:")
print(df[['Player Name', 'Team', 'Runs', 'HS', 'HS_numeric', 'AVG', 'AVG_numeric']].head(10).to_string())

# =============================================================================
# STEP 4 — STATISTICAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("[STEP 4] Statistical Summary")
print("=" * 60)

numeric_cols = ['Runs', 'Matches', 'Inn', 'BF', 'SR', '100s', '50s', '4s', '6s']
print(df[numeric_cols].describe().round(2).to_string())

print("\nHighlights:")
print(f"  Highest Runs    : {df['Runs'].max()} — {df.loc[df['Runs'].idxmax(), 'Player Name']}")
print(f"  Average Runs    : {df['Runs'].mean():.1f}")
print(f"  Median Runs     : {df['Runs'].median():.1f}")
print(f"  Best SR Overall : {df['SR'].max()} — {df.loc[df['SR'].idxmax(), 'Player Name']}")
print(f"  Most Sixes      : {df['6s'].max()} — {df.loc[df['6s'].idxmax(), 'Player Name']}")
print(f"  Most Fours      : {df['4s'].max()} — {df.loc[df['4s'].idxmax(), 'Player Name']}")

# =============================================================================
# Q1 — TOP 10 RUN SCORERS
# =============================================================================
print("\n" + "=" * 60)
print("[Q1] Who are the Top 10 Run Scorers of IPL 2025?")
print("=" * 60)

top10 = df.nlargest(10, 'Runs')[['Player Name', 'Team', 'Runs', 'Matches', 'AVG_numeric', 'SR']].reset_index(drop=True)
top10.index += 1
top10.columns = ['Player', 'Team', 'Runs', 'Matches', 'Average', 'Strike Rate']
print(top10.to_string())
print("\n📌 Insight: Sai Sudharsan (GT) topped with 759 runs.")
print("   PBKS had 2 players (Shreyas Iyer & Prabhsimran Singh) in top 10.")

# =============================================================================
# Q2 — TEAM TOTAL RUNS
# =============================================================================
print("\n" + "=" * 60)
print("[Q2] Which Team Scored the Most Total Runs?")
print("=" * 60)

team_stats = df.groupby('Team').agg(
    Total_Runs = ('Runs',        'sum'),
    Players    = ('Player Name', 'count'),
    Centuries  = ('100s',        'sum'),
    Fifties    = ('50s',         'sum'),
    Total_6s   = ('6s',          'sum'),
    Total_4s   = ('4s',          'sum'),
).sort_values('Total_Runs', ascending=False)

print(team_stats.to_string())
print(f"\n📌 Insight: PBKS led with 3000 runs. KKR scored the least (1886).")

# =============================================================================
# Q3 — BEST BATTING AVERAGE (min 5 matches)
# =============================================================================
print("\n" + "=" * 60)
print("[Q3] Who Has the Best Batting Average? (min 5 matches)")
print("=" * 60)

qualified = df[(df['Matches'] >= 5) & (df['AVG_numeric'].notna())].copy()
best_avg = qualified.nlargest(10, 'AVG_numeric')[['Player Name', 'Team', 'AVG_numeric', 'Runs', 'Matches']].reset_index(drop=True)
best_avg.index += 1
best_avg.columns = ['Player', 'Team', 'Average', 'Runs', 'Matches']
print(best_avg.to_string())
print(f"\n📌 Insight: Surya Kumar Yadav leads with avg of 65.18 among qualified players.")

# =============================================================================
# Q4 — BEST STRIKE RATE (min 100 balls)
# =============================================================================
print("\n" + "=" * 60)
print("[Q4] Who Has the Best Strike Rate? (min 100 balls faced)")
print("=" * 60)

sr_q = df[df['BF'] >= 100].nlargest(10, 'SR')[['Player Name', 'Team', 'SR', 'Runs', 'BF']].reset_index(drop=True)
sr_q.index += 1
sr_q.columns = ['Player', 'Team', 'Strike Rate', 'Runs', 'Balls Faced']
print(sr_q.to_string())
print(f"\n📌 Insight: Vaibhav Suryavanshi (RR) had SR of 206.55 — best in IPL 2025.")

# =============================================================================
# Q5 — RUN DISTRIBUTION
# =============================================================================
print("\n" + "=" * 60)
print("[Q5] How Are Runs Distributed Across All Players?")
print("=" * 60)

bins   = [0, 50, 100, 200, 300, 400, 500, 800]
labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '500+']
df['Run_Category'] = pd.cut(df['Runs'], bins=bins, labels=labels)

cat_counts = df['Run_Category'].value_counts().sort_index()
for cat, count in cat_counts.items():
    bar = '█' * count
    pct = count / len(df) * 100
    print(f"  {cat:<10} : {bar} ({count} players, {pct:.1f}%)")

print(f"\n📌 Insight: {cat_counts['0-50'] + cat_counts['51-100']} players scored under 100 runs.")
print("   Top-order batsmen dominate run scoring in T20 cricket.")

# =============================================================================
# Q6 — CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[Q6] Correlation Between Batting Metrics")
print("=" * 60)

numeric_df = df[['Runs', 'Matches', 'Inn', 'BF', 'SR', '100s', '50s', '4s', '6s', 'AVG_numeric']].copy()
corr = numeric_df.corr().round(2)
print(corr.to_string())

print(f"\n📌 Key Correlations:")
print(f"  Runs vs Balls Faced    : {corr.loc['Runs', 'BF']}  (very strong — more balls = more runs)")
print(f"  Runs vs Fours          : {corr.loc['Runs', '4s']}  (strong — boundaries drive the score)")
print(f"  Runs vs Innings        : {corr.loc['Runs', 'Inn']}  (more innings = more chances)")
print(f"  Strike Rate vs Average : {corr.loc['SR', 'AVG_numeric']}  (moderate positive relationship)")

# =============================================================================
# Q7 — TEAM BOUNDARIES
# =============================================================================
print("\n" + "=" * 60)
print("[Q7] Which Team Hit the Most Sixes and Fours?")
print("=" * 60)

boundary_stats = df.groupby('Team').agg(
    Total_4s = ('4s', 'sum'),
    Total_6s = ('6s', 'sum')
).reset_index()
boundary_stats['Total_Boundaries'] = boundary_stats['Total_4s'] + boundary_stats['Total_6s']
boundary_stats = boundary_stats.sort_values('Total_Boundaries', ascending=False).reset_index(drop=True)
boundary_stats.index += 1
print(boundary_stats.to_string())

top_6s_team = boundary_stats.nlargest(1, 'Total_6s').iloc[0]
top_4s_team = boundary_stats.nlargest(1, 'Total_4s').iloc[0]
print(f"\n📌 Most Sixes: {top_6s_team['Team']} ({int(top_6s_team['Total_6s'])} sixes)")
print(f"   Most Fours: {top_4s_team['Team']} ({int(top_4s_team['Total_4s'])} fours)")

# =============================================================================
# Q8 — CENTURIES AND FIFTIES
# =============================================================================
print("\n" + "=" * 60)
print("[Q8] Who Scored the Most Fifties and Centuries?")
print("=" * 60)

print("\nTop 10 by Fifties:")
top_50s = df.nlargest(10, '50s')[['Player Name', 'Team', '50s', '100s', 'Runs']].reset_index(drop=True)
top_50s.index += 1
print(top_50s.to_string())

print("\nCentury Scorers:")
centurions = df[df['100s'] > 0][['Player Name', 'Team', '100s', 'HS', 'Runs']].sort_values('Runs', ascending=False).reset_index(drop=True)
centurions.index += 1
print(centurions.to_string())
print(f"\n📌 Only {len(centurions)} players scored a century in IPL 2025.")

# =============================================================================
# Q9 — TEAM AVERAGE STRIKE RATE
# =============================================================================
print("\n" + "=" * 60)
print("[Q9] Average Strike Rate Per Team (min 20 balls faced)")
print("=" * 60)

sr_team = df[df['BF'] >= 20].groupby('Team')['SR'].mean().round(2).sort_values(ascending=False).reset_index()
sr_team.columns = ['Team', 'Avg Strike Rate']
sr_team.index += 1
print(sr_team.to_string())
print(f"\n📌 {sr_team.iloc[0]['Team']} had the highest avg SR: {sr_team.iloc[0]['Avg Strike Rate']}")

# =============================================================================
# Q10 — INNINGS vs RUNS
# =============================================================================
print("\n" + "=" * 60)
print("[Q10] Does Playing More Innings Lead to More Runs?")
print("=" * 60)

corr_inn = df['Inn'].corr(df['Runs']).round(3)
corr_bf  = df['BF'].corr(df['Runs']).round(3)
print(f"  Innings vs Runs     : {corr_inn}")
print(f"  Balls Faced vs Runs : {corr_bf}")

df['Inn_Group'] = pd.cut(df['Inn'], bins=[0,3,6,9,12,17], labels=['1-3','4-6','7-9','10-12','13+'])
inn_group = df.groupby('Inn_Group')['Runs'].mean().round(1).reset_index()
inn_group.columns = ['Innings Range', 'Avg Runs']
print()
print(inn_group.to_string())
print(f"\n📌 Clear trend: Players with 13+ innings average {inn_group.iloc[-1]['Avg Runs']} runs")
print(f"   vs only {inn_group.iloc[0]['Avg Runs']} runs for players with 1-3 innings.")

# =============================================================================
# BONUS — TOP SCORER PER TEAM
# =============================================================================
print("\n" + "=" * 60)
print("[BONUS] Top Scorer for Each Team")
print("=" * 60)

top_per_team = df.loc[df.groupby('Team')['Runs'].idxmax()][['Team', 'Player Name', 'Runs', 'AVG', 'SR']]
top_per_team = top_per_team.sort_values('Runs', ascending=False).reset_index(drop=True)
top_per_team.index += 1
print(top_per_team.to_string())

# =============================================================================
# SAVE CLEANED DATASET
# =============================================================================
df_clean = df.copy()
df_clean.to_csv('IPL2025_Cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved → IPL2025_Cleaned.csv")
print("   (Use this file for Task 3 - Data Visualization)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  ✅  TASK 2 — EDA COMPLETE!")
print("=" * 60)
print()
print("  Key Findings:")
print("  1. Sai Sudharsan (GT) — Top scorer with 759 runs")
print("  2. PBKS — Highest team total (3000 runs)")
print("  3. Surya Kumar Yadav — Best average (65.18)")
print("  4. Vaibhav Suryavanshi — Best SR (206.55, min 100 balls)")
print("  5. 60%+ players scored under 200 runs")
print("  6. Runs vs Balls Faced — strongest correlation (0.99)")
print("  7. LSG hit the most sixes (152)")
print("  8. Only 9 players scored a century in IPL 2025")
print("  9. PBKS had the highest avg team strike rate")
print(" 10. More innings = significantly more runs")
print()
print("  Next → Task 3: Data Visualization")
print("=" * 60)
