/**
 * @file merge_datasets.C
 * @brief Merges raw .root files together, keeping only selected branches.
 */
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TClass.h>
#include <TFileMerger.h>
#include <iostream>
#include <string>
#include <vector>


/**
 * @brief Returns the name of all TTree in the file
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
 * @brief Ordinal suffix helpers for branch naming (st, nd, rd, fourth, fifth, sixth)
 */
const std::vector<std::string> kJetSuffixes = {
    "st", "nd", "rd", "fourth", "fifth", "sixth"
};
const int kNJetSlots = 6;


/**
 * @brief Main.
 */
void data_preparation()
{
    const std::string file1  = "../OriginalTrainingDatasets/DYJetsToLL/4578E947-084C-C946-9B8D-1B45A126DCED.root";
    const std::string file2  = "../OriginalTrainingDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root";
    const std::string file3  = "../OriginalTrainingDatasets/ZZTo2L2Nu/DC33D4B8-4AF1-C94A-8F03-EDB634488D2B.root";
    const std::string output = "../TrainingDataset/training.root";

    const std::vector<std::string> inputFiles = {file1, file2, file3};

    // Jet array variables that will be unpacked into N scalar branches
    const std::vector<std::string> jetVecBranches = {
        "Jet_pt", "Jet_phi", "Jet_eta", "Jet_mass"
    };

    // Selecting scalar branches to save
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

    auto treeNames = GetTreeNames(file1);
    if (treeNames.empty()) {
        std::cerr << "[ERROR] No TTree found in " << file1 << std::endl;
        return;
    }

    bool firstTree = true;

    for (const auto& treeName : treeNames) {
        // Processing only the TTree "Events"
        if (treeName != "Events") {
            std::cout << "[INFO] Skipping TTree: " << treeName << std::endl;
            continue;
        }

        std::cout << "[INFO] Processing TTree: " << treeName << std::endl;

        ROOT::RDataFrame rdf_raw(treeName, inputFiles);

        // Expanding each Jet vector branch into kNJetSlots scalar columns
        // with zero-padding when nJet < kNJetSlots.
        ROOT::RDF::RNode rdf = rdf_raw;

        // Collect the names of every new scalar column we generate
        std::vector<std::string> expandedCols;

        for (const auto& vecBranch : jetVecBranches) {

            // Check the vector branch actually exists in this tree
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
                const int slot = i; // capture by value for the lambda

                // ROOT::RVec<float> is the typical NanoAOD type;
                // adjust to double if your files use doubles.
                rdf = rdf.Define(newCol,
                    [slot](const ROOT::RVecF& v) -> float {
                        return (slot < (int)v.size()) ? v[slot] : 0.f;
                    },
                    {vecBranch}
                );

                expandedCols.push_back(newCol);
            }
        }

        // Build the final list of branches to snapshot:
        //   1. scalar selectedBranches that exist in the tree
        //   2. the newly defined scalar jet columns
        auto availableCols = rdf.GetColumnNames();

        std::vector<std::string> validBranches;

        // original scalar branches
        for (const auto& b : selectedBranches) {
            if (std::find(availableCols.begin(), availableCols.end(), b)
                    != availableCols.end()) {
                validBranches.push_back(b);
            } else {
                std::cout << "  [WARN] Branch not found, skipped: " << b << std::endl;
            }
        }

        // expanded jet scalar columns (already guaranteed to exist via Define)
        for (const auto& col : expandedCols)
            validBranches.push_back(col);

        if (validBranches.empty()) {
            std::cerr << "[ERROR] No valid branch for: " << treeName << std::endl;
            continue;
        }

        // Quality filters
        rdf = rdf.Filter("MET_pt > 0", "MET_pt > 0");
        rdf = rdf.Filter("Jet_pt_st > 0", "Leading jet pt > 0");
        rdf = rdf.Filter("Jet_mass_st > 0", "Leading jet mass > 0");

        ROOT::RDF::RSnapshotOptions opts;
        opts.fMode             = firstTree ? "RECREATE" : "UPDATE";
        opts.fLazy             = false;
        opts.fCompressionLevel = 1;

        auto report = rdf.Report();
        rdf.Snapshot(treeName, output, validBranches, opts);

        auto nEntries = rdf.Count().GetValue();
        std::cout << " -> " << nEntries << " written entries" << std::endl;

        // Prints the cutflow
        report->Print();

        firstTree = false;
    }

    // Copy non-TTree objects
    std::cout << "[INFO] Copying non-TTree objects" << std::endl;

    TFileMerger merger(kFALSE);
    merger.OutputFile(output.c_str(), "UPDATE");
    for (const auto& f : inputFiles)
        merger.AddFile(f.c_str());
    merger.SetNotrees(kTRUE);
    merger.Merge();

    std::cout << "\nMerging completed. Output: " << output << std::endl;
}