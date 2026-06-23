/**
 * @file testing_data_preparation.C
 * @brief Keeps only selected branches for the testing datasets.
 */
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TClass.h>
#include <iostream>
#include <string>
#include <vector>


/**
 * @brief Returns the name of all TTrees in the file.
 */
std::vector<std::string> GetTreeNames(const std::string& filename)
{
    std::vector<std::string> names;
    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) return names;
    TIter next(f->GetListOfKeys());
    TKey* key;
    while ((key = (TKey*)next())) {
        TClass* cl = TClass::GetClass(key->GetClassName());
        if (cl && cl->InheritsFrom("TTree"))
            names.push_back(key->GetName());
    }
    f->Close();
    return names;
}


/**
 * @brief Ordinal suffix helpers for Jets branch naming.
 */
const std::vector<std::string> kJetSuffixes = {
    "st", "nd", "rd", "fourth", "fifth", "sixth"
};
const int kNJetSlots = 6;


/**
 * @brief Main.
 */
void testing_data_preparation(){
    const std::string input = "../../OriginalTestingDatasets/HToAATo2Mu2B/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root";
    const std::string output = "../../TestingDatasets/testing_HToAATo2Mu2B.root";

    const std::vector<std::string> jetVecBranches = {
        "Jet_pt", "Jet_phi", "Jet_eta", "Jet_mass"
    };

    const std::vector<std::string> selectedBranches = {
        "GenMET_pt",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "MET_phi",
        "MET_pt",
        "MET_significance",
        "MET_sumEt",
        "MET_sumPtUnclustered",
        "fixedGridRhoFastjetAll",
        "fixedGridRhoFastjetCentral",
        "fixedGridRhoFastjetCentralCalo",
        "fixedGridRhoFastjetCentralChargedPileUp",
        "fixedGridRhoFastjetCentralNeutral",
        "PV_ndof",
        "PV_x",
        "PV_y",
        "PV_z",
        "PV_chi2",
        "PV_score",
        "PV_npvs",
        "PV_npvsGood",
        "nSV",
        "nJet",
    };

    auto treeNames = GetTreeNames(input);
    if (treeNames.empty()) {
        std::cerr << "[ERROR] No TTree found in " << input << std::endl;
        return;
    }

    bool firstTree = true;

    /**
     * @brief Collects scalar branches and processes the vectorial ones.
     * Expands each Jet vector branch into kNJetSlots scalar columns with zero-padding when nJet < kNJetSlots.
     */
    for (const auto& treeName : treeNames) {
        if (treeName != "Events") {
            std::cout << "[INFO] Skipping TTree: " << treeName << std::endl;
            continue;
        }

        std::cout << "[INFO] Processing TTree: " << treeName << std::endl;

        ROOT::RDataFrame rdf_raw(treeName, input);

        ROOT::RDF::RNode rdf = rdf_raw;

        // Expand jet vector branches into scalar slots
        std::vector<std::string> expandedCols;

        for (const auto& vecBranch : jetVecBranches) {
            auto availableCheck = rdf.GetColumnNames();
            bool vecExists = std::find(
                availableCheck.begin(), availableCheck.end(), vecBranch
            ) != availableCheck.end();

            if (!vecExists) {
                std::cout << "  [WARN] Vector branch not found, skipping expansion: "
                          << vecBranch << std::endl;
                continue;
            }

            for (int i = 0; i < kNJetSlots; ++i) {
                const std::string newCol = vecBranch + "_" + kJetSuffixes[i];
                const int slot = i;  // capture by value for the lambda

                rdf = rdf.Define(newCol,
                    [slot](const ROOT::RVecF& v) -> float {
                        return (slot < (int)v.size()) ? v[slot] : 0.f;
                    },
                    {vecBranch}
                );

                expandedCols.push_back(newCol);
            }
        }

        /**
         * @brief Builds the final list of branches to snapshot:
         * 1. scalar selectedBranches that exist in the tree;
         * 2. newly defined scalar jet columns.
         */
        auto availableCols = rdf.GetColumnNames();

        std::vector<std::string> validBranches;

        // Original scalar branches
        for (const auto& b : selectedBranches) {
            if (std::find(availableCols.begin(), availableCols.end(), b)
                    != availableCols.end()) {
                validBranches.push_back(b);
            } else {
                std::cout << "  [WARN] Branch not found, skipped: " << b << std::endl;
            }
        }

        // Expanded jet scalar columns
        for (const auto& col : expandedCols)
            validBranches.push_back(col);

        if (validBranches.empty()) {
            std::cerr << "[ERROR] No valid branch for: " << treeName << std::endl;
            continue;
        }

        /**
         * @brief Setting quality filters.
         */
        rdf = rdf.Filter("MET_pt > 0", "MET_pt > 0");
        rdf = rdf.Filter("Jet_pt_st > 0", "Leading jet pt > 0");
        rdf = rdf.Filter("Jet_mass_st > 0", "Leading jet mass > 0");

        /**
         * @brief Cleaning snapshot.
         */
        ROOT::RDF::RSnapshotOptions opts;
        opts.fMode             = firstTree ? "RECREATE" : "UPDATE";
        opts.fLazy             = false;
        opts.fCompressionLevel = 1;

        auto report = rdf.Report();
        rdf.Snapshot(treeName, output, validBranches, opts);
        report->Print();

        TFile* fout = TFile::Open(output.c_str(), "READ");
        TTree* tout = fout->Get<TTree>(treeName.c_str());
        std::cout << " -> " << tout->GetEntries() << " written entries" << std::endl;
        fout->Close();

        firstTree = false;
    }

    std::cout << "\nTesting data preparation completed. Output: " << output << std::endl;
}