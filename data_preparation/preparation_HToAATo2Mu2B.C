#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>


using RNode = ROOT::RDF::RNode;
template<typename T> using RVec = ROOT::RVec<T>;
using LV = ROOT::Math::PtEtaPhiMVector;


/**
 * @brief Prints dataset statistics:
 * - number of total events;
 * - maximum number of jets and muons;
 * - maximum number of secondary vertices;
 * - mean and standard deviation of GenMET_pt.
 */
void print_stats(ROOT::RDataFrame& df) {
    auto n_tot         = df.Count();
    auto max_njet      = df.Max<UInt_t>("nJet");
    auto max_nmuon     = df.Max<UInt_t>("nMuon");
    auto max_nsv       = df.Max<UInt_t>("nSV");
    auto mean_genmet   = df.Mean<Float_t>("GenMET_pt");
    auto stddev_genmet = df.StdDev<Float_t>("GenMET_pt");

    std::cout << "nEvents before skimming:  " << *n_tot         << "\n"
              << "Max nJet:                 " << *max_njet      << "\n"
              << "Max nMuon:                " << *max_nmuon     << "\n"
              << "Max nSV:                  " << *max_nsv       << "\n"
              << "Mean GenMET_pt:           " << *mean_genmet   << "\n"
              << "StdDev GenMET_pt:         " << *stddev_genmet << "\n";
}


/**
 * @brief Helper function.
 * Finds the two indices with higher btag score.
 */
std::pair<size_t, size_t> twoMaxBtagIndices(const RVec<Float_t>& btag) {
    size_t i1 = 0, i2 = 1;
    if (btag[i2] > btag[i1]) std::swap(i1, i2);
    for (size_t i = 2; i < btag.size(); i++) {
        if (btag[i] > btag[i1]) { i2 = i1; i1 = i; }
        else if (btag[i] > btag[i2]) { i2 = i; }
    }
    return {i1, i2};
}


/**
 * @brief Computes delta phi in [-pi, pi].
 */
Float_t deltaPhi(Float_t phi1, Float_t phi2) {
    Float_t dphi = phi1 - phi2;
    while (dphi > M_PI) dphi -= 2 * M_PI;
    while (dphi < -M_PI) dphi += 2 * M_PI;
    return dphi;
}


/**
 * @brief Helper function.
 * Assigns all the properties of the jets to the best jet or second best
 * according to the highest btag score.
 */
RNode define_jet_columns(RNode df) {

    // Computes the two indices
    df = df.Define("_btag_indices",
        [](const RVec<Float_t>& btag) -> RVec<Int_t> {
            auto [i1, i2] = twoMaxBtagIndices(btag);
            return {(Int_t)i1, (Int_t)i2};
        }, {"Jet_btagDeepFlavB"});

    // Btag scores
    df = df
        .Define("Jet_btag_bst", [](const RVec<Float_t>& b, const RVec<Int_t>& idx)
            { return b[idx[0]]; }, {"Jet_btagDeepFlavB", "_btag_indices"})
        .Define("Jet_btag_bnd", [](const RVec<Float_t>& b, const RVec<Int_t>& idx)
            { return b[idx[1]]; }, {"Jet_btagDeepFlavB", "_btag_indices"});

    // Assigns all other jet properties
    auto def_jet = [&](RNode n,
                       const std::string& prop,
                       const std::string& branch) -> RNode {
        return n
            .Define("Jet_" + prop + "_bst",
                [](const RVec<Float_t>& v, const RVec<Int_t>& idx){ return v[idx[0]]; },
                {branch, "_btag_indices"})
            .Define("Jet_" + prop + "_bnd",
                [](const RVec<Float_t>& v, const RVec<Int_t>& idx){ return v[idx[1]]; },
                {branch, "_btag_indices"});
    };

    df = def_jet(df, "eta",       "Jet_eta");
    df = def_jet(df, "pt",        "Jet_pt");
    df = def_jet(df, "phi",       "Jet_phi");
    df = def_jet(df, "mass",      "Jet_mass");
    df = def_jet(df, "area",      "Jet_area");
    df = def_jet(df, "rawFactor", "Jet_rawFactor");
    df = def_jet(df, "chHEF",     "Jet_chHEF");
    df = def_jet(df, "neHEF",     "Jet_neHEF");
    df = def_jet(df, "chEmEF",    "Jet_chEmEF");
    df = def_jet(df, "neEmEF",    "Jet_neEmEF");
    df = def_jet(df, "muEF",      "Jet_muEF");

    return df;
}


/**
 * @brief Defines muons columns.
 */
RNode define_muon_columns(RNode df) {
    return df
        .Define("Muon_charge_st", [](const RVec<Int_t>&   v){ return v[0]; }, {"Muon_charge"})
        .Define("Muon_dxy_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_dxy"})
        .Define("Muon_dz_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_dz"})
        .Define("Muon_eta_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_eta"})
        .Define("Muon_mass_st",   [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_mass"})
        .Define("Muon_phi_st",    [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_phi"})
        .Define("Muon_pt_st",     [](const RVec<Float_t>& v){ return v[0]; }, {"Muon_pt"})
        .Define("Muon_charge_nd", [](const RVec<Int_t>&   v){ return v[1]; }, {"Muon_charge"})
        .Define("Muon_dxy_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_dxy"})
        .Define("Muon_dz_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_dz"})
        .Define("Muon_eta_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_eta"})
        .Define("Muon_mass_nd",   [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_mass"})
        .Define("Muon_phi_nd",    [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_phi"})
        .Define("Muon_pt_nd",     [](const RVec<Float_t>& v){ return v[1]; }, {"Muon_pt"});
}


/**
 * @brief Finds the best secondary vertex.
 * @param dlenSig: decay lengh significance (distance score between PV and SV).
 */
RNode define_sv_columns(RNode df) {
    return df
        .Define("SV_dlenSig_bst",
            [](UInt_t nSV, const RVec<Float_t>& dlenSig) -> Float_t {
                if (nSV == 0) return 0.f;
                return *std::max_element(dlenSig.begin(), dlenSig.begin() + nSV);
            }, {"nSV", "SV_dlenSig"})
        .Define("SV_mass_bst",
            [](UInt_t nSV, const RVec<Float_t>& dlenSig,
               const RVec<Float_t>& mass) -> Float_t {
                if (nSV == 0) return 0.f;
                size_t best = std::max_element(dlenSig.begin(), dlenSig.begin() + nSV) - dlenSig.begin();
                return mass[best];
            }, {"nSV", "SV_dlenSig", "SV_mass"});
}


/**
 * @brief Defines new Branches for the engineered features.
 */
RNode define_engineered_columns(RNode df) {

    // Fourvectors
    df = df
        .Define("_lv_mu1", [](Float_t pt, Float_t eta, Float_t phi, Float_t m)
            { return LV(pt, eta, phi, m); },
            {"Muon_pt_st", "Muon_eta_st", "Muon_phi_st", "Muon_mass_st"})
        .Define("_lv_mu2", [](Float_t pt, Float_t eta, Float_t phi, Float_t m)
            { return LV(pt, eta, phi, m); },
            {"Muon_pt_nd", "Muon_eta_nd", "Muon_phi_nd", "Muon_mass_nd"})
        .Define("_lv_jet1", [](Float_t pt, Float_t eta, Float_t phi, Float_t m)
            { return LV(pt, eta, phi, m); },
            {"Jet_pt_bst", "Jet_eta_bst", "Jet_phi_bst", "Jet_mass_bst"})
        .Define("_lv_jet2", [](Float_t pt, Float_t eta, Float_t phi, Float_t m)
            { return LV(pt, eta, phi, m); },
            {"Jet_pt_bnd", "Jet_eta_bnd", "Jet_phi_bnd", "Jet_mass_bnd"})
        .Define("_lv_mumu",    [](LV a, LV b){ return a + b; }, {"_lv_mu1",  "_lv_mu2"})
        .Define("_lv_bb",      [](LV a, LV b){ return a + b; }, {"_lv_jet1", "_lv_jet2"})
        .Define("_lv_mumu_bb", [](LV a, LV b){ return a + b; }, {"_lv_mumu", "_lv_bb"});

    // Invariants masses
    df = df
        .Define("M_mumu",    [](LV v){ return Float_t(v.M()); }, {"_lv_mumu"})
        .Define("M_bb",      [](LV v){ return Float_t(v.M()); }, {"_lv_bb"})
        .Define("M_mumu_bb", [](LV v){ return Float_t(v.M()); }, {"_lv_mumu_bb"});

    // DeltaR between MET direction and bb system (conventionally MET_eta=0)
    df = df.Define("dR_MET_bb",
        [](Float_t met_phi, LV bb) -> Float_t {
            Float_t deta = 0.f - bb.Eta();
            Float_t dphi = deltaPhi(met_phi, bb.Phi());
            return std::sqrt(deta*deta + dphi*dphi);
        }, {"MET_phi", "_lv_bb"});

    // MET projections along and perpendicular to the bb axis
    df = df
        .Define("MET_projection_par",
            [](Float_t met_pt, Float_t met_phi, LV bb) -> Float_t {
                return met_pt * std::cos(deltaPhi(met_phi, (Float_t)bb.Phi()));
            }, {"MET_pt", "MET_phi", "_lv_bb"})
        .Define("MET_projection_perp",
            [](Float_t met_pt, Float_t met_phi, LV bb) -> Float_t {
                return met_pt * std::sin(deltaPhi(met_phi, (Float_t)bb.Phi()));
            }, {"MET_pt", "MET_phi", "_lv_bb"});

    // Delta phi between MET and leading objects
    df = df
        .Define("dPhi_MET_mu1",
            [](Float_t met_phi, Float_t mu_phi) -> Float_t {
                return deltaPhi(met_phi, mu_phi);
            }, {"MET_phi", "Muon_phi_st"})
        .Define("dPhi_MET_jet1",
            [](Float_t met_phi, Float_t jet_phi) -> Float_t {
                return deltaPhi(met_phi, jet_phi);
            }, {"MET_phi", "Jet_phi_bst"});

    // HT: scalar sum over all the jets pt
    df = df.Define("HT",
        [](UInt_t nJet, const RVec<Float_t>& pt) -> Float_t {
            Float_t ht = 0.f;
            for (UInt_t j = 0; j < nJet; j++) ht += pt[j];
            return ht;
        }, {"nJet", "Jet_pt"});

    return df;
}


/**
 * @brief Main.
 */
void preparation_HToAATo2Mu2B() {

    /**
     * @brief Builds the RDataFrame.
     */
    ROOT::RDataFrame df("Events",
        "../OriginalDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");

    /**
     * @brief Prints the dataset stats.
     */
    print_stats(df);

    /**
     * @brief Defines cinematic filters.
     */
    auto df_sel = df
        .Filter("MET_pt >= 0", "MET_pt >= 0")
        .Filter("nMuon >= 2", "nMuon >= 2")
        .Filter("nJet >= 2", "nJet >= 2");

    /**
     * @brief Defines the new branches.
     */
    auto df_def = define_jet_columns(df_sel);
    df_def = define_muon_columns(df_def);
    df_def = define_sv_columns(df_def);

    /**
     * @brief Applies filters for the physics of the process.
     */
    const Float_t btag_threshold = 0.5f;
    df_def = df_def
        .Filter([btag_threshold](Float_t b1, Float_t b2){
                    return b1 > btag_threshold && b2 > btag_threshold; },
                {"Jet_btag_bst", "Jet_btag_bnd"}, "btag threshold")
        .Filter("Jet_pt_bst > 0 && Jet_pt_bnd > 0", "Jet pt > 0")
        .Filter("Muon_pt_st > 0 && Muon_pt_nd > 0", "Muon pt > 0")
        .Filter([](const RVec<Int_t>& charge){ return charge[0] + charge[1] == 0; },
        {"Muon_charge"}, "OS muon pair");

    /**
     * @brief Feature engineering.
     */
    df_def = define_engineered_columns(df_def);

    /**
     * @brief Saves all Branches.
     */
    const std::vector<std::string> output_cols = {
        // Scalar features
        "MET_pt", "MET_phi", "MET_covXX", "MET_covXY", "MET_covYY",
        "MET_significance", "GenMET_pt",
        "PV_chi2", "PV_score", "PV_x", "PV_y", "PV_z",
        "nSV", "nJet", "nMuon", "nGenJet", "fixedGridRhoFastjetAll",
        // Jets
        "Jet_eta_bst",  "Jet_pt_bst",  "Jet_phi_bst",  "Jet_mass_bst",
        "Jet_area_bst", "Jet_btag_bst","Jet_rawFactor_bst",
        "Jet_chHEF_bst","Jet_neHEF_bst","Jet_chEmEF_bst","Jet_neEmEF_bst","Jet_muEF_bst",
        "Jet_eta_bnd",  "Jet_pt_bnd",  "Jet_phi_bnd",  "Jet_mass_bnd",
        "Jet_area_bnd", "Jet_btag_bnd","Jet_rawFactor_bnd",
        "Jet_chHEF_bnd","Jet_neHEF_bnd","Jet_chEmEF_bnd","Jet_neEmEF_bnd","Jet_muEF_bnd",
        // Muons
        "Muon_charge_st","Muon_dxy_st","Muon_dz_st","Muon_eta_st",
        "Muon_mass_st",  "Muon_phi_st","Muon_pt_st",
        "Muon_charge_nd","Muon_dxy_nd","Muon_dz_nd","Muon_eta_nd",
        "Muon_mass_nd",  "Muon_phi_nd","Muon_pt_nd",
        // SV
        "SV_dlenSig_bst", "SV_mass_bst",
        // Feature engineering
        "M_mumu", "M_bb", "M_mumu_bb",
        "dR_MET_bb", "MET_projection_par", "MET_projection_perp",
        "dPhi_MET_mu1", "dPhi_MET_jet1", "HT"
    };

    /**
     * @brief Snapshots the results of the skimming.
     */
    const std::string outfile = "../CleanedDatasets/cleaned_HToAATo2Mu2B.root";
    df_def.Snapshot("Events", outfile, output_cols);

    // Final report
    std::cout << "\n    Dataset preparation report    \n";
    df.Report()->Print();
}