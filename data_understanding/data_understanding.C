/**
 * @file data_understanding.C
 * @brief Takes only single value branches do perform data understanding
 * over the different datasets.
 */
#include <ROOT/RDataFrame.hxx>
#include <iostream>
#include <string>
#include <vector>


/**
 * @brief List of single-value branches to keep in the skimmed dataset.
 */
const std::vector<std::string> Branches = {
    // target
    "GenMET_pt",
    // MET reconstructed
    "MET_pt",
    "MET_covXX",
    "MET_covXY",
    "MET_covYY",
    "MET_phi",
    "MET_significance",
    "MET_sumEt",
    "MET_sumPtUnclustered",
    "MET_MetUnclustEnUpDeltaX",
    "MET_MetUnclustEnUpDeltaY",
    // Pileup info + generated Jets (only on MC simulations)
    "Pileup_nTrueInt",
    "Pileup_pudensity",
    "Pileup_gpudensity",
    "Pileup_nPU",
    "Pileup_sumEOOT",
    "Pileup_sumLOOT",
    "nGenJet",
    // Energy density
    "fixedGridRhoFastjetAll",
    "fixedGridRhoFastjetCentral",
    "fixedGridRhoFastjetCentralCalo",
    "fixedGridRhoFastjetCentralChargedPileUp",
    "fixedGridRhoFastjetCentralNeutral",
    // Primary vertex
    "PV_ndof",
    "PV_x",
    "PV_y",
    "PV_z",
    "PV_chi2",
    "PV_score",
    "PV_npvs",
    "PV_npvsGood",
    // Object multiplicity
    "nSV",
    "nJet",
};


/**
 * @brief Main.
 */
void data_understanding() {
    /**
     * @brief Builds RDataFrame directly from the TChain-equivalent:
     * - first argument is the TTree name
     * - second is the input file (or a vector/glob of files for multiple inputs).
     */
    ROOT::RDataFrame df(
        "Events",
        "../OriginalTrainingDatasets/DYJetsToLL/4578E947-084C-C946-9B8D-1B45A126DCED.root"
    );

    /**
     * @brief Prints total number of events.
     */
    auto n_events = df.Count();

    /**
     * @brief Writes the skimmed dataset.
     * Snapshot selects only the branches in Branches and writes
     * them to the output file. The event loop runs here.
     */
    df.Snapshot("Events",
                "UnderstandingDatasets/understanding_DYJetsToLL.root",
                Branches);

    std::cout << "nEvents: " << *n_events << std::endl;
}