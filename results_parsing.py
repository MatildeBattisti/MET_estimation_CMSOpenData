import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Heatmap for Pearson correlations
"""
pearson_corr = pd.read_csv("results/ZZTo2L2Nu_results/skimmed/pearson_corr.csv", index_col=0)
plt.figure(figsize=(12,10))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Pearson Correlation Heatmap for the skimmed dataset")
plt.tight_layout()
plt.show()

pearson_corr_specific = pd.read_csv("results/ZZTo2L2Nu_results/specific_skimmed/pearson_corr.csv", index_col=0)
target = "GenMET_pt"

pearson_corr = pearson_corr_specific.corr(method='pearson')[target].drop(target)
pearson_corr = pearson_corr.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=pearson_corr.index, y=pearson_corr.values, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Pearson Correlation with GenMET_pt")
plt.title("Pearson correlation for the specific feature selection")
plt.tight_layout()
plt.show()

"""
    Heatmap for Spearman correlations
"""
spearman_corr = pd.read_csv("results/ZZTo2L2Nu_results/skimmed/spearman_corr.csv", index_col=0)
plt.figure(figsize=(12,10))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Spearman Correlation Heatmap")
plt.tight_layout()
plt.show()

spearman_corr_specific = pd.read_csv("results/ZZTo2L2Nu_results/specific_skimmed/spearman_corr.csv", index_col=0)

spearman_corr = spearman_corr_specific.corr(method='spearman')[target].drop(target)
spearman_corr = spearman_corr.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=spearman_corr.index, y=spearman_corr.values, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Spearman Correlation with GenMET_pt")
plt.title("Spearman correlation for the specific feature selection")
plt.tight_layout()
plt.show()

"""
    Scatterplot for GenMET_pt vs MET_pt
"""
skimmed_MET = pd.read_json("results/ZZTo2L2Nu_results/skimmed/MET_vs_GenMET.json")
specific_skimmed_MET = pd.read_json("results/ZZTo2L2Nu_results/specific_skimmed/MET_vs_GenMET.json")

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)

# Skimmed
axes[0].scatter(skimmed_MET["GenMET_pt"], skimmed_MET["MET_pt"], alpha=0.3, color="blue")
axes[0].plot([skimmed_MET["GenMET_pt"].min(), skimmed_MET["GenMET_pt"].max()],
             [skimmed_MET["GenMET_pt"].min(), skimmed_MET["GenMET_pt"].max()],
             'r--', lw=1)
axes[0].set_xlabel("GenMET_pt")
axes[0].set_ylabel("MET_pt")
axes[0].set_title("Dataset skimmed with generic features")

# Specific skimmed
axes[1].scatter(specific_skimmed_MET["GenMET_pt"], specific_skimmed_MET["MET_pt"], alpha=0.3, color="green")
axes[1].plot([specific_skimmed_MET["GenMET_pt"].min(), specific_skimmed_MET["GenMET_pt"].max()],
             [specific_skimmed_MET["GenMET_pt"].min(), specific_skimmed_MET["GenMET_pt"].max()],
             'r--', lw=1)
axes[1].set_xlabel("GenMET_pt")
axes[1].set_title("Dataset skimmed with specifcally chosen features")

plt.suptitle("GenMET vs MET")
plt.tight_layout()
plt.show()

"""
    Scatterplot for predicted MET vs GenMET_pt
"""
scatter_test = pd.read_json("results/ZZTo2L2Nu_results/skimmed/METpred_vs_GenMET.json")
scatter_test_specific = pd.read_json("results/ZZTo2L2Nu_results/specific_skimmed/METpred_vs_GenMET.json")

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)

# Skimmed
axes[0].scatter(scatter_test["y_test"], scatter_test["y_test_pred"], alpha=0.3, color="blue")
axes[0].plot([scatter_test["y_test"].min(), scatter_test["y_test"].max()],
             [scatter_test["y_test"].min(), scatter_test["y_test"].max()],
             'r--', lw=1)
axes[0].set_xlabel("GenMET_pt [Gev]")
axes[0].set_ylabel("Predicted MET [Gev]")
axes[0].set_title("Dataset skimmed with generic features")

# Specific skimmed
axes[1].scatter(scatter_test_specific["y_test"], scatter_test_specific["y_test_pred"], alpha=0.3, color="green")
axes[1].plot([scatter_test_specific["y_test"].min(), scatter_test_specific["y_test"].max()],
             [scatter_test_specific["y_test"].min(), scatter_test_specific["y_test"].max()],
             'r--', lw=1)
axes[1].set_xlabel("GenMET_pt [Gev]")
axes[1].set_title("Dataset skimmed with specifcally chosen features")

plt.suptitle("GenMET vs PredictedMET")
plt.tight_layout()
plt.show()

"""
    Statistics to evaluate the goodness of the model
"""
skimmed_stats = pd.read_csv("results/ZZTo2L2Nu_results/skimmed/summary_stats.csv")
stats_dict = skimmed_stats.iloc[0].to_dict()
print("Stats for the dataset skimmed with generic entries:")
for key, value in stats_dict.items():
    print(f"{key}: {value}")

skimmed_stats_specific = pd.read_csv("results/ZZTo2L2Nu_results/specific_skimmed/summary_stats.csv")
stats_dict_specific = skimmed_stats_specific.iloc[0].to_dict()
print("\nStats for the dataset skimmed with specific entries:")
for key, value in stats_dict_specific.items():
    print(f"{key}: {value}")




