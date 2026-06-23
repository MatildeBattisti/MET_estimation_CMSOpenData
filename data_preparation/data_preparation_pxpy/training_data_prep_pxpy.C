/**
 * @file training_data_prep_pxpy.C
 * @brief Merges raw .root files together, keeping only selected branches.
 *        MET, GenMET and Jet quantities are stored as px/py instead of pt/phi.
 *        Draws kSamplesPerFile random events from each input file for balance.
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
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_set>
#include <atomic>
#include <memory>
#include <cmath>


/**
 * @brief Sets per file samples number.
 * Set to -1 for no sampling.
 */
const Long64_t kSamplesPerFile = 40000;


/**
 * @brief Gets the TTree's names.
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
 * @brief Gets the number of entries.
 */
Long64_t GetNEntries(const std::string& filename, const std::string& treeName)
{
    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) return -1;
    TTree* t = (TTree*)f->Get(treeName.c_str());
    Long64_t n = t ? t->GetEntries() : -1;
    f->Close();
    return n;
}


/**
 * @brief Assigns random indices to each entry of the dataset.
 */
std::vector<Long64_t> RandomIndices(Long64_t nTotal, Long64_t kSamples,
                                    unsigned int seed = 42)
{
    if (kSamples < 0 || kSamples >= nTotal) {
        std::vector<Long64_t> all(nTotal);
        std::iota(all.begin(), all.end(), 0LL);
        return all;
    }
    std::mt19937_64 rng(seed);
    std::vector<Long64_t> idx(nTotal);
    std::iota(idx.begin(), idx.end(), 0LL);
    for (Long64_t i = 0; i < kSamples; ++i) {
        std::uniform_int_distribution<Long64_t> dist(i, nTotal - 1);
        std::swap(idx[i], idx[dist(rng)]);
    }
    idx.resize(kSamples);
    std::sort(idx.begin(), idx.end());
    return idx;
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
void training_data_prep_pxpy(){
    const std::string file1  = "../../OriginalTrainingDatasets/DYJetsToLL/4578E947-084C-C946-9B8D-1B45A126DCED.root";
    const std::string file2  = "../../OriginalTrainingDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root";
    const std::string file3  = "../../OriginalTrainingDatasets/ZZTo2L2Nu/DC33D4B8-4AF1-C94A-8F03-EDB634488D2B.root";
    const std::string output = "../../TrainingDataset/training_pxpy.root";

    const std::vector<std::string> inputFiles = {file1, file2, file3};

    const std::vector<std::string> jetVecBranches = {
        "Jet_eta", "Jet_mass", "Jet_px", "Jet_py"
    };

    const std::vector<std::string> selectedBranches = {
        "GenMET_px",
        "GenMET_py",
        "MET_px",
        "MET_py",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
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

    std::vector<std::string> tmpFiles;
    for (std::size_t i = 0; i < inputFiles.size(); ++i)
        tmpFiles.push_back("../../TrainingDataset/tmp_pxpy_" + std::to_string(i) + ".root");

    /**
     * @brief Per-file atomic counter loop (defined once, reset before each file)
     * to select random entries.
     */
    auto counterPtr = std::make_shared<std::atomic<Long64_t>>(-1);

    for (std::size_t fileIdx = 0; fileIdx < inputFiles.size(); ++fileIdx) {
        const std::string& inFile  = inputFiles[fileIdx];
        const std::string& tmpFile = tmpFiles[fileIdx];

        std::cout << "\n[INFO] File " << (fileIdx + 1) << "/" << inputFiles.size()
                  << ": " << inFile << std::endl;

        bool firstTreeInFile = true;

        for (const auto& treeName : treeNames) {
            if (treeName != "Events") {
                std::cout << "[INFO] Skipping TTree: " << treeName << std::endl;
                continue;
            }
            std::cout << "[INFO] Processing TTree: " << treeName << std::endl;

            // Random index set
            Long64_t nTotal = GetNEntries(inFile, treeName);
            if (nTotal <= 0) {
                std::cerr << "  [ERROR] Could not read entry count.\n";
                continue;
            }
            std::cout << "  Total entries available : " << nTotal << "\n";

            Long64_t nDraw = (kSamplesPerFile < 0 || kSamplesPerFile >= nTotal)
                             ? nTotal : kSamplesPerFile;
            std::cout << "  Drawing " << nDraw << " random entries "
                      << "(seed=" << (42 + (unsigned)fileIdx) << ")\n";

            auto chosen = RandomIndices(nTotal, nDraw, 42 + (unsigned)fileIdx);
            auto indexSetPtr = std::make_shared<std::unordered_set<Long64_t>>(
                chosen.begin(), chosen.end());

            // Reset counter to -1 before processing next file
            counterPtr->store(-1);

            // Build RDataFrame and apply random index filter
            ROOT::RDataFrame rdf_raw(treeName, inFile);
            ROOT::RDF::RNode rdf = rdf_raw
                .Define("__entry_idx__",
                    [counterPtr]() -> Long64_t {
                        return counterPtr->fetch_add(1);
                    }, {})
                .Filter(
                    [indexSetPtr](Long64_t idx) {
                        return indexSetPtr->count(idx) > 0;
                    },
                    {"__entry_idx__"}, "Random index selection");

            /**
             * @brief Feature engineering.
             * Transforming pt, phi into px, py:
             * px = pt * cos(phi),  py = pt * sin(phi)
             */
            // Scalar cartesian components for GenMET and MET
            rdf = rdf.Define("GenMET_px",
                [](float pt, float phi) { return pt * std::cos(phi); },
                {"GenMET_pt", "GenMET_phi"}
            );
            rdf = rdf.Define("GenMET_py",
                [](float pt, float phi) { return pt * std::sin(phi); },
                {"GenMET_pt", "GenMET_phi"}
            );
            rdf = rdf.Define("MET_px",
                [](float pt, float phi) { return pt * std::cos(phi); },
                {"MET_pt", "MET_phi"}
            );
            rdf = rdf.Define("MET_py",
                [](float pt, float phi) { return pt * std::sin(phi); },
                {"MET_pt", "MET_phi"}
            );

            // Jet_px and Jet_py as RVec columns
            rdf = rdf.Define("Jet_px",
                [](const ROOT::RVecF& pt, const ROOT::RVecF& phi) {
                    return ROOT::RVecF(pt * cos(phi));
                },
                {"Jet_pt", "Jet_phi"}
            );
            rdf = rdf.Define("Jet_py",
                [](const ROOT::RVecF& pt, const ROOT::RVecF& phi) {
                    return ROOT::RVecF(pt * sin(phi));
                },
                {"Jet_pt", "Jet_phi"}
            );

            // Expand jet vector branches into scalar slots
            std::vector<std::string> expandedCols;
            
            for (const auto& vecBranch : jetVecBranches) {
                auto cols = rdf.GetColumnNames();
                if (std::find(cols.begin(), cols.end(), vecBranch) == cols.end()) {
                    std::cout << "  [WARN] Branch not found, skipping: "
                              << vecBranch << "\n";
                    continue;
                }
                for (int i = 0; i < kNJetSlots; ++i) {
                    const std::string newCol = vecBranch + "_" + kJetSuffixes[i];
                    const int slot = i;
                    rdf = rdf.Define(newCol,
                        [slot](const ROOT::RVecF& v) -> float {
                            return (slot < (int)v.size()) ? v[slot] : 0.f;
                        }, {vecBranch});
                    expandedCols.push_back(newCol);
                }
            }

            auto availableCols = rdf.GetColumnNames();
            std::vector<std::string> validBranches;
            for (const auto& b : selectedBranches) {
                if (std::find(availableCols.begin(), availableCols.end(), b)
                        != availableCols.end())
                    validBranches.push_back(b);
                else
                    std::cout << "  [WARN] Branch not found, skipped: " << b << "\n";
            }
            for (const auto& col : expandedCols)
                validBranches.push_back(col);

            if (validBranches.empty()) {
                std::cerr << "  [ERROR] No valid branches for " << treeName << "\n";
                continue;
            }

            /**
             * @brief Sets quality filters
             */
            rdf = rdf.Filter("MET_pt > 0", "MET_pt > 0");
            rdf = rdf.Filter(
                [](const ROOT::RVecF& pt) {
                    return !pt.empty() && pt[0] > 0.f;
                },
                {"Jet_pt"}, "Leading jet pt > 0"
            );
            rdf = rdf.Filter("Jet_mass_st > 0", "Leading jet mass > 0");

            /**
             * @brief Snapshot to temporary file.
             */
            ROOT::RDF::RSnapshotOptions opts;
            opts.fMode             = firstTreeInFile ? "RECREATE" : "UPDATE";
            opts.fLazy             = false;
            opts.fCompressionLevel = 1;

            auto report = rdf.Report();
            rdf.Snapshot(treeName, tmpFile, validBranches, opts);

            auto nWritten = rdf.Count().GetValue();
            std::cout << "  -> " << nWritten << " entries written to " << tmpFile << "\n";
            report->Print();

            firstTreeInFile = false;
        }
    }

    /**
     * @brief Merges temporary files into final output.
     */
    std::cout << "\n[INFO] Merging temporary files into " << output << std::endl;
    {
        TFileMerger merger(kFALSE);
        merger.OutputFile(output.c_str(), "RECREATE");
        for (const auto& tmp : tmpFiles)
            merger.AddFile(tmp.c_str());
        merger.Merge();
    }

    /**
     * @brief Copies non-TTree objects from original inputs.
     */
    std::cout << "[INFO] Copying non-TTree objects from original inputs\n";
    {
        TFileMerger merger(kFALSE);
        merger.OutputFile(output.c_str(), "UPDATE");
        for (const auto& f : inputFiles)
            merger.AddFile(f.c_str());
        merger.SetNotrees(kTRUE);
        merger.Merge();
    }

    /**
     * @brief Removes temporary files.
     */
    std::cout << "[INFO] Removing temporary files\n";
    for (const auto& tmp : tmpFiles) {
        if (std::remove(tmp.c_str()) != 0)
            std::cerr << "  [WARN] Could not remove " << tmp << "\n";
        else
            std::cout << "  Removed " << tmp << "\n";
    }

    std::cout << "\nDone. Balanced output written to: " << output << std::endl;
}