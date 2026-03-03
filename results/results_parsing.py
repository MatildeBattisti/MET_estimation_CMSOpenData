import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Statistics to evaluate the goodness of the model
"""
skimmed_stats = pd.read_csv("results_XGB/HToAATo2Mu2B/skimmed/summary_stats.csv")
stats_dict = skimmed_stats.iloc[0].to_dict()
print("Stats for the dataset skimmed with common features:\n")
for key, value in stats_dict.items():
    print(f"{key}: {value}")

skimmed_stats_specific = pd.read_csv("results_XGB/HToAATo2Mu2B/specific_skimmed/summary_stats.csv")
stats_dict_specific = skimmed_stats_specific.iloc[0].to_dict()
print("\n\nStats for the dataset skimmed with specific features:\n")
for key, value in stats_dict_specific.items():
    print(f"{key}: {value}")

"""
    Heatmap for Pearson correlations
"""
pearson_corr = pd.read_csv("results_XGB/HToAATo2Mu2B/skimmed/pearson_corr.csv", index_col=0)
plt.figure(figsize=(12,10))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Pearson Correlation heatmap for the common features")
plt.tight_layout()
plt.show()

pearson_corr_specific = pd.read_csv("results_XGB/HToAATo2Mu2B/specific_skimmed/pearson_corr.csv", index_col=0)
target = "GenMET_pt"

pearson_corr = pearson_corr_specific.corr(method='pearson')[target].drop(target)
pearson_corr = pearson_corr.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=pearson_corr.index, y=pearson_corr.values, hue=pearson_corr.index,
            palette="coolwarm", legend=False)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Pearson Correlation with GenMET_pt")
plt.title("Pearson correlation for the specific feature selection")
plt.tight_layout()
plt.show()

"""
    Heatmap for Spearman correlations
"""
spearman_corr = pd.read_csv("results_XGB/HToAATo2Mu2B/skimmed/spearman_corr.csv", index_col=0)
plt.figure(figsize=(12,10))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Spearman Correlation heatmap for the common features")
plt.tight_layout()
plt.show()

spearman_corr_specific = pd.read_csv("results_XGB/HToAATo2Mu2B/specific_skimmed/spearman_corr.csv", index_col=0)

spearman_corr = spearman_corr_specific.corr(method='spearman')[target].drop(target)
spearman_corr = spearman_corr.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=spearman_corr.index, y=spearman_corr.values, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Spearman Correlation with GenMET_pt")
plt.title("Spearman correlation for the specific feature selection")
plt.tight_layout()
plt.show()

"""
    Scaling of the evaluation
"""
result_df = pd.read_json("results_XGB/HToAATo2Mu2B/skimmed/retraining_log.json")
plt.plot(result_df["rmse"], label="Validation RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("XGBoost Validation RMSE")
plt.legend()
plt.show()

"""
    Scatterplot for GenMET_pt vs MET_pt
"""
skimmed_MET = pd.read_json("results_XGB/HToAATo2Mu2B/skimmed/MET_vs_GenMET.json")
specific_skimmed_MET = pd.read_json("results_XGB/HToAATo2Mu2B/specific_skimmed/MET_vs_GenMET.json")

fig1, axes1 = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)

# Skimmed
axes1[0].scatter(skimmed_MET["GenMET_pt"], skimmed_MET["MET_pt"], alpha=0.3, color="blue")
axes1[0].plot([skimmed_MET["GenMET_pt"].min(), skimmed_MET["GenMET_pt"].max()],
              [skimmed_MET["GenMET_pt"].min(), skimmed_MET["GenMET_pt"].max()],
              'r--', lw=1)
axes1[0].set_xlabel("GenMET_pt [GeV]")
axes1[0].set_ylabel("MET_pt [GeV]")
axes1[0].set_title("Dataset skimmed with common features")

# Specific skimmed
axes1[1].scatter(specific_skimmed_MET["GenMET_pt"], specific_skimmed_MET["MET_pt"], alpha=0.3, color="green")
axes1[1].plot([specific_skimmed_MET["GenMET_pt"].min(), specific_skimmed_MET["GenMET_pt"].max()],
              [specific_skimmed_MET["GenMET_pt"].min(), specific_skimmed_MET["GenMET_pt"].max()],
              'r--', lw=1)
axes1[1].set_xlabel("GenMET_pt [GeV]")
axes1[1].set_title("Dataset skimmed with specific features")

fig1.suptitle("GenMET vs MET")
fig1.tight_layout()

"""
    Scatterplot for predicted MET vs GenMET_pt
"""
scatter_test = pd.read_json("results_XGB/HToAATo2Mu2B/skimmed/METpred_vs_GenMET.json")
scatter_test_specific = pd.read_json("results_XGB/HToAATo2Mu2B/specific_skimmed/METpred_vs_GenMET.json")

fig2, axes2 = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)

# Skimmed
axes2[0].scatter(scatter_test["y_test"], scatter_test["y_test_pred"], alpha=0.3, color="blue")
axes2[0].plot([scatter_test["y_test"].min(), scatter_test["y_test"].max()],
              [scatter_test["y_test"].min(), scatter_test["y_test"].max()],
              'r--', lw=1)
axes2[0].set_xlabel("GenMET_pt [GeV]")
axes2[0].set_ylabel("Predicted MET [GeV]")
axes2[0].set_title("Dataset skimmed with common features")

# Specific skimmed
axes2[1].scatter(scatter_test_specific["y_test"], scatter_test_specific["y_test_pred"], alpha=0.3, color="green")
axes2[1].plot([scatter_test_specific["y_test"].min(), scatter_test_specific["y_test"].max()],
              [scatter_test_specific["y_test"].min(), scatter_test_specific["y_test"].max()],
              'r--', lw=1)
axes2[1].set_xlabel("GenMET_pt [GeV]")
axes2[1].set_title("Dataset skimmed with specific features")

fig2.suptitle("GenMET vs PredictedMET")
fig2.tight_layout()

plt.show()





