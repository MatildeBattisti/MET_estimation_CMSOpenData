#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>


using RNode = ROOT::RDF::RNode;
template<typename T> using RVec = ROOT::RVec<T>;


/**
 * @brief Prints dataset statistics:
 * - number of total events;
 * - maximum number of leptons;
 * - mean and standard deviation of GenMET_pt.
 */
void print_stats(ROOT::RDataFrame& df) {
    auto n_tot = df.Count();
    auto max_ne = df.Max<UInt_t>("nElectron");
    auto max_nm = df.Max<UInt_t>("nMuon");
    auto mean_genmet = df.Mean<Float_t>("GenMET_pt");
    auto stddev_genmet = df.StdDev<Float_t>("GenMET_pt");

    std::cout << "nEvents before skimming:          " << *n_tot        << "\n"
              << "Max nElectron:                    " << *max_ne       << "\n"
              << "Max nMuon:                        " << *max_nm       << "\n"
              << "Mean GenMET_pt:                   " << *mean_genmet  << "\n"
              << "StdDev GenMET_pt:                 " << *stddev_genmet << "\n";
}


/**
 * @brief Helper function for the event loop.
 * Adds the electrons columns to the node while keeping the muons columns
 * empty (2e + 0mu) channel.
 */
RNode define_ee_columns(RNode df) {
    return df
        // First electron
        .Define("Electron_charge_st", [](const RVec<Int_t>&   v){ return v[0]; }, {"Electron_charge"})
        .Define("Electron_dxy_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_dxy"})
        .Define("Electron_dz_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_dz"})
        .Define("Electron_eta_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_eta"})
        .Define("Electron_mass_st",   [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_mass"})
        .Define("Electron_phi_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_phi"})
        .Define("Electron_pt_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Electron_pt"})
        // Second electron
        .Define("Electron_charge_nd", [](const RVec<Int_t>&   v){ return v[1]; }, {"Electron_charge"})
        .Define("Electron_dxy_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_dxy"})
        .Define("Electron_dz_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_dz"})
        .Define("Electron_eta_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_eta"})
        .Define("Electron_mass_nd",   [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_mass"})
        .Define("Electron_phi_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_phi"})
        .Define("Electron_pt_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Electron_pt"})
        // Empty muons
        .Define("Muon_charge_st",  []{ return Int_t{0}; })
        .Define("Muon_charge_nd",  []{ return Int_t{0}; })
        .Define("Muon_dxy_st",     []{ return Float_t{0}; })
        .Define("Muon_dxy_nd",     []{ return Float_t{0}; })
        .Define("Muon_dz_st",      []{ return Float_t{0}; })
        .Define("Muon_dz_nd",      []{ return Float_t{0}; })
        .Define("Muon_eta_st",     []{ return Float_t{0}; })
        .Define("Muon_eta_nd",     []{ return Float_t{0}; })
        .Define("Muon_mass_st",    []{ return Float_t{0}; })
        .Define("Muon_mass_nd",    []{ return Float_t{0}; })
        .Define("Muon_phi_st",     []{ return Float_t{0}; })
        .Define("Muon_phi_nd",     []{ return Float_t{0}; })
        .Define("Muon_pt_st",      []{ return Float_t{0}; })
        .Define("Muon_pt_nd",      []{ return Float_t{0}; });
}


/**
 * @brief Helper function for the event loop.
 * Adds the muons columns to the node while keeping the electrons columns
 * empty (0e + 2mu) channel.
 */
RNode define_mumu_columns(RNode df) {
    return df
        // First muon
        .Define("Muon_charge_st", [](const RVec<Int_t>&   v){ return v[0]; }, {"Muon_charge"})
        .Define("Muon_dxy_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_dxy"})
        .Define("Muon_dz_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_dz"})
        .Define("Muon_eta_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_eta"})
        .Define("Muon_mass_st",   [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_mass"})
        .Define("Muon_phi_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_phi"})
        .Define("Muon_pt_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_pt"})
        // Second muon
        .Define("Muon_charge_nd", [](const RVec<Int_t>&   v){ return v[1]; }, {"Muon_charge"})
        .Define("Muon_dxy_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_dxy"})
        .Define("Muon_dz_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_dz"})
        .Define("Muon_eta_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_eta"})
        .Define("Muon_mass_nd",   [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_mass"})
        .Define("Muon_phi_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_phi"})
        .Define("Muon_pt_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_pt"})
        // Empty electrons
        .Define("Electron_charge_st", []{ return Int_t{0}; })
        .Define("Electron_charge_nd", []{ return Int_t{0}; })
        .Define("Electron_dxy_st",    []{ return Float_t{0}; })
        .Define("Electron_dxy_nd",    []{ return Float_t{0}; })
        .Define("Electron_dz_st",     []{ return Float_t{0}; })
        .Define("Electron_dz_nd",     []{ return Float_t{0}; })
        .Define("Electron_eta_st",    []{ return Float_t{0}; })
        .Define("Electron_eta_nd",    []{ return Float_t{0}; })
        .Define("Electron_mass_st",   []{ return Float_t{0}; })
        .Define("Electron_mass_nd",   []{ return Float_t{0}; })
        .Define("Electron_phi_st",    []{ return Float_t{0}; })
        .Define("Electron_phi_nd",    []{ return Float_t{0}; })
        .Define("Electron_pt_st",     []{ return Float_t{0}; })
        .Define("Electron_pt_nd",     []{ return Float_t{0}; });
}


/**
 * @brief Main.
 */
void preparation_ZZTo2L2Nu() {

    /**
     * @brief Builds the RDataFrame.
     */
    ROOT::RDataFrame df("Events",
        "../OriginalDatasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");

    /**
     * @brief Prints the dataset stats.
     */
    print_stats(df);

    /**
     * @brief Saves the single value input Branches.
     */
    const std::vector<std::string> scalar_cols = {
        "nElectron", "nMuon",
        "MET_pt", "MET_phi", "MET_covXX", "MET_covXY", "MET_covYY",
        "MET_significance", "GenMET_pt",
        "PV_chi2", "PV_score", "PV_x", "PV_y", "PV_z",
        "nSV"
    };

    /**
     * @brief Saves the new output Branches.
     */
    const std::vector<std::string> lepton_cols = {
        "Electron_charge_st", "Electron_charge_nd",
        "Electron_dxy_st",    "Electron_dxy_nd",
        "Electron_dz_st",     "Electron_dz_nd",
        "Electron_eta_st",    "Electron_eta_nd",
        "Electron_mass_st",   "Electron_mass_nd",
        "Electron_phi_st",    "Electron_phi_nd",
        "Electron_pt_st",     "Electron_pt_nd",
        "Muon_charge_st",     "Muon_charge_nd",
        "Muon_dxy_st",        "Muon_dxy_nd",
        "Muon_dz_st",         "Muon_dz_nd",
        "Muon_eta_st",        "Muon_eta_nd",
        "Muon_mass_st",       "Muon_mass_nd",
        "Muon_phi_st",        "Muon_phi_nd",
        "Muon_pt_st",         "Muon_pt_nd"
    };

    
    /**
     * @brief Merges scalar and leptons columns.
     */
    std::vector<std::string> output_cols = scalar_cols;
    output_cols.insert(output_cols.end(), lepton_cols.begin(), lepton_cols.end());

    // ee channel
    auto df_ee = define_ee_columns(
        df.Filter("nElectron == 2 && nMuon == 0", "2e + 0mu")
    );
    auto df_ee_clean = df_ee
        .Filter([](Float_t m1, Float_t m2){ return m1 >= 0.f && m2 >= 0.f; },
                {"Electron_mass_st", "Electron_mass_nd"},
                "Electron mass >= 0")
        .Filter("MET_pt >= 0", "MET_pt >= 0")
        .Filter("Electron_pt_st > 0 && Electron_pt_nd > 0", "Electron pt > 0")
        .Filter("Electron_charge_st + Electron_charge_nd == 0", "OS electron pair");

    // mumu channel
    auto df_mumu = define_mumu_columns(
        df.Filter("nMuon == 2 && nElectron == 0", "0e + 2mu")
    );
    auto df_mumu_clean = df_mumu
        .Filter([](Float_t m1, Float_t m2){ return m1 >= 0.f && m2 >= 0.f; },
                {"Muon_mass_st", "Muon_mass_nd"},
                "Muon mass >= 0")
        .Filter("MET_pt >= 0", "MET_pt >= 0")
        .Filter("Muon_pt_st > 0 && Muon_pt_nd > 0", "Muon pt > 0")
        .Filter("Muon_charge_st + Muon_charge_nd == 0", "OS muon pair");

    
    /**
     * @brief Snapshots the results of the skimming.
     */
    const std::string outfile = "../CleanedDatasets/cleaned_ZZTo2L2Nu.root";

    // Writes the channels on two temporary separate files
    df_ee_clean.Snapshot("Events", "tmp_ee.root", output_cols);
    df_mumu_clean.Snapshot("Events", "tmp_mumu.root", output_cols);

    // Merges the two temporary files
    TChain merged("Events");
    merged.Add("tmp_ee.root");
    merged.Add("tmp_mumu.root");
    merged.Merge(outfile.c_str());

    // Temporary files removal
    if (std::remove("tmp_ee.root") != 0)
        std::cerr << "Warning: cannot remove tmp_ee.root\n";
    if (std::remove("tmp_mumu.root") != 0)
        std::cerr << "Warning: cannot remove tmp_mumu.root\n";

    // Final report
    std::cout << "\n    Dataset preparation report    \n";
    df.Report()->Print();
}