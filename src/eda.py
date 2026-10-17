import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/animal_disease_dataset.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df.info()
df.describe(include = "all")

print(((df['age'] > 0) & (df['age'] <= 15)).sum()) 
#all ages are between 1 - 15 years

print(((df['temperature'] >= 100) & (df['temperature'] <= 105)).sum())
#all temperatures are between 100 - 105 degrees Fahrenheit

print(f"NaN values in each column:\n{df.isnull().sum()}")
#no NaN values in the dataset

print(f"Duplicate rows in the dataset:\n{df.duplicated().sum()}")
#no duplicate rows in the dataset

print(f"Number of rows for each disease:\n{df['disease'].value_counts()}")

# sns.countplot(data=df, x='disease')
# plt.title("Class balance")
# plt.tight_layout()
# plt.savefig("reports/figures/class_balance.png", dpi=150)
# plt.close()

symptom_cols = ["symptom_1", "symptom_2", "symptom_3"]


#overall symptom frequency (all three symptom columns combined)
symptoms = df[symptom_cols].stack()
counts = symptoms.value_counts()
print(counts)

# plt.figure(figsize=(10, 8))
# sns.barplot(x=counts.values, y=counts.index)
# plt.title("Symptom frequency (all three symptom columns combined)")
# plt.xlabel("Count")
# plt.ylabel("Symptom")
# plt.tight_layout()
# plt.savefig("reports/figures/symptom_counts.png", dpi=150)
# plt.close()

# symptom vs disease heatmap
long = df.melt(id_vars="disease", value_vars=symptom_cols, value_name="symptom")
table = pd.crosstab(long["symptom"], long["disease"], normalize="columns")
print(table.round(2))

# plt.figure(figsize=(10, 10))
# sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues")
# plt.title("Share of each disease's symptom mentions, by symptom")
# plt.tight_layout()
# plt.savefig("reports/figures/symptom_vs_disease_heatmap.png", dpi=150)
# plt.close()


df['symptom_set'] = df[symptom_cols].apply(frozenset, axis = 1)
print("rows with fewer than 3 distinct symptoms:", (df["symptom_set"].apply(len) < 3).sum())

print(f"distinct symptom sets: {df['symptom_set'].nunique()}")
diseases_per_set = df.groupby("symptom_set")['disease'].nunique()
print(diseases_per_set.value_counts().sort_index())

shared = diseases_per_set[diseases_per_set > 1].index
rows_in_shared = df["symptom_set"].isin(shared).sum()
print("sets shared by >1 disease:", len(shared))
print("rows in shared sets:", rows_in_shared, f"({rows_in_shared/len(df):.1%})")

shared_df = df[df["symptom_set"].isin(shared)]
table = pd.crosstab(shared_df["symptom_set"].apply(lambda s: ", ".join(sorted(s))), shared_df["disease"])
print(table.to_string())

ceiling = df.groupby("symptom_set")["disease"].agg(lambda x: x.value_counts().iloc[0]).sum() / len(df)
print(f"symptom-only accuracy ceiling: {ceiling:.1%}")


print(df.groupby("disease")[["age", "temperature"]].agg(["mean", "std", "min", "max"]).round(2))

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="disease", y="age")
plt.title("Age by disease")
plt.tight_layout()
plt.savefig("reports/figures/age_by_disease.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="disease", y="temperature")
plt.title("Temperature (°F) by disease")
plt.tight_layout()
plt.savefig("reports/figures/temperature_by_disease.png", dpi=150)
plt.close()

print(pd.crosstab(df["animal"], df["disease"]))
print(pd.crosstab(df["animal"], df["disease"], normalize="columns").round(3))

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="animal", hue="disease")
plt.title("Disease counts by animal")
plt.tight_layout()
plt.savefig("reports/figures/animal_vs_disease.png", dpi=150)
plt.close()

hard = df[df["disease"].isin(["lumpy virus", "pneumonia"])]
print(hard.groupby("disease")[["age", "temperature"]].describe().round(2).T)
print(pd.crosstab(hard["animal"], hard["disease"], normalize="columns").round(3))

plt.figure(figsize=(8, 5))
sns.boxplot(data=hard, x="disease", y="temperature")
plt.title("Temperature: lumpy virus vs pneumonia")
plt.tight_layout()
plt.savefig("reports/figures/temp_lumpy_vs_pneumonia.png", dpi=150)
plt.close()

# ---------- Observations ----------
# 1. 43,778 rows, 7 columns, no missing values, no duplicate rows.
# 2. Age is 1-15 and temperature is 100-105 F, so all values are valid.
# 3. 5 diseases, mildly imbalanced (anthrax is the largest, lumpy virus the smallest).
# 4. 82 distinct symptom sets. 81 map to exactly one disease.
# 5. Lumpy virus and pneumonia share one symptom set (depression, loss of appetite,
#    painless lumps), covering 33.2% of rows. Symptom-only accuracy ceiling is 83.6%.
# 6. Age, temperature and animal look the same across all diseases, so they do not
#    separate lumpy virus from pneumonia either.
# 7. The dataset is synthetic.