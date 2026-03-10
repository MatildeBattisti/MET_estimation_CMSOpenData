#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <cmath>
#include <stdexcept>



/**
 * @brief Defines single value input global variables:
 * - the entries have name and type of the dataset Branches;
 * - the entries Branch status is set to one;
 * - gets the address of the Branches in order to copy their values.
 */
struct Branches {
    Float_t MET_pt;
    Float_t MET_phi;
    Float_t MET_covXX;
    Float_t MET_covXY;
    Float_t MET_covYY;
    Float_t MET_significance;
    Float_t GenMET_pt;
    Float_t PV_chi2;
    Float_t PV_score;
    Float_t PV_x;
    Float_t PV_y;
    Float_t PV_z;
    UInt_t nSV;
    UInt_t nJet;
    UInt_t nMuon;
    UInt_t nGenJet;
    Float_t fixedGridRhoFastjetAll;

    void setup_branches(TChain* chain) {
        chain->SetBranchStatus("*", 0);

        chain->SetBranchStatus("MET_pt", 1);
        chain->SetBranchStatus("MET_phi", 1);
        chain->SetBranchStatus("MET_covXX", 1);
        chain->SetBranchStatus("MET_covXY", 1);
        chain->SetBranchStatus("MET_covYY", 1);
        chain->SetBranchStatus("MET_significance", 1);
        chain->SetBranchStatus("GenMET_pt", 1);
        chain->SetBranchStatus("PV_chi2", 1);
        chain->SetBranchStatus("PV_score", 1);
        chain->SetBranchStatus("PV_x", 1);
        chain->SetBranchStatus("PV_y", 1);
        chain->SetBranchStatus("PV_z", 1);
        chain->SetBranchStatus("nSV", 1);
        chain->SetBranchStatus("nJet", 1);
        chain->SetBranchStatus("nMuon", 1);
        chain->SetBranchStatus("nGenJet", 1);
        chain->SetBranchStatus("fixedGridRhoFastjetAll", 1);

        chain->SetBranchAddress("MET_pt", &MET_pt);
        chain->SetBranchAddress("MET_phi", &MET_phi);
        chain->SetBranchAddress("MET_covXX", &MET_covXX);
        chain->SetBranchAddress("MET_covXY", &MET_covXY);
        chain->SetBranchAddress("MET_covYY", &MET_covYY);
        chain->SetBranchAddress("MET_significance", &MET_significance);
        chain->SetBranchAddress("GenMET_pt", &GenMET_pt);
        chain->SetBranchAddress("PV_chi2", &PV_chi2);
        chain->SetBranchAddress("PV_score", &PV_score);
        chain->SetBranchAddress("PV_x", &PV_x);
        chain->SetBranchAddress("PV_y", &PV_y);
        chain->SetBranchAddress("PV_z", &PV_z);
        chain->SetBranchAddress("nSV", &nSV);
        chain->SetBranchAddress("nJet", &nJet);
        chain->SetBranchAddress("nMuon", &nMuon);
        chain->SetBranchAddress("nGenJet", &nGenJet);
        chain->SetBranchAddress("fixedGridRhoFastjetAll", &fixedGridRhoFastjetAll);
    }
};



/**
 * @brief Gets the maximum number of jets, muons and secondary vertices.
 */
struct MaxMultiplicities {
    UInt_t nJet;
    UInt_t nMuon;
    UInt_t nSV;
};

MaxMultiplicities getMaxMultiplicities(TChain* chain, Long64_t n_events,
                                        UInt_t& nJet, UInt_t& nMuon, UInt_t& nSV) {
    MaxMultiplicities max = {0, 0, 0};
    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (nJet  > max.nJet) max.nJet = nJet;
        if (nMuon > max.nMuon) max.nMuon = nMuon;
        if (nSV   > max.nSV) max.nSV = nSV;
    }
    return max;
}



/**
 * @brief Calculates mean and standard deviation of GenMET_pt.
 */
std::pair<Float_t, Float_t> computeMeanAndStdGenMET(TChain* chain, Long64_t n_events, Float_t& GenMET_pt) {
    // Mean
    Float_t sum = 0.0;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        sum += GenMET_pt;
    }
    Float_t GenMET_mean = sum / n_events;

    // Standard deviation
    Float_t GenMET_variance_num = 0.0;
    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        GenMET_variance_num += (GenMET_pt - GenMET_mean) * (GenMET_pt - GenMET_mean);
    }
    Float_t GenMET_variance = GenMET_variance_num / n_events;
    Float_t GenMET_std = std::sqrt(GenMET_variance);

    return {GenMET_mean, GenMET_std};
}



/**
 * @brief Finds the two indeces with higher btag score.
 */
std::pair<size_t, size_t> findTwoMaxIndices(const std::vector<Float_t>& arr, UInt_t n) {
    if (n < 2) {
        throw std::invalid_argument("The array has to have at least two arguments.");
    }

    size_t maxIdx = 0;
    size_t secondMaxIdx = 1;

    if (arr[secondMaxIdx] > arr[maxIdx]) {
        std::swap(maxIdx, secondMaxIdx);
    }

    for (size_t i = 2; i < n; i++) {
        if (arr[i] > arr[maxIdx]) {
            secondMaxIdx = maxIdx;
            maxIdx = i;
        } else if (arr[i] > arr[secondMaxIdx]) {
            secondMaxIdx = i;
        }
    }

    return {maxIdx, secondMaxIdx};
}



/**
 * @brief Computes delta phi in [-pi, pi].
 */
Float_t deltaPhi(Float_t phi1, Float_t phi2) {
    Float_t dphi = phi1 - phi2;
    while (dphi >  M_PI) dphi -= 2 * M_PI;
    while (dphi < -M_PI) dphi += 2 * M_PI;
    return dphi;
}



/**
 * @brief Computes delta R between two objects.
 */
Float_t deltaR(Float_t eta1, Float_t phi1, Float_t eta2, Float_t phi2) {
    Float_t deta = eta1 - eta2;
    Float_t dphi = deltaPhi(phi1, phi2);
    return std::sqrt(deta * deta + dphi * dphi);
}



/**
 * @brief Builds a TLorentzVector from pt, eta, phi, mass.
 */
TLorentzVector buildLV(Float_t pt, Float_t eta, Float_t phi, Float_t mass) {
    TLorentzVector lv;
    lv.SetPtEtaPhiM(pt, eta, phi, mass);
    return lv;
}



void preparation_HToAATo2Mu2B() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("../OriginalDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");



    /**
     * @param n_events Number of events in each file.
     */
    Long64_t n_events = chain->GetEntries();

    std::cout << "nEvents before skimming:" << n_events << std::endl;



    /**
     * @brief Sets up the branches.
     */
    Branches branches;
    branches.setup_branches(chain.get());



    /**
     * @brief Gets the maximum number of jet, muon, SV arrays.
     */
    auto max = getMaxMultiplicities(chain.get(), n_events,
                                    branches.nJet, branches.nMuon, branches.nSV);

    std::cout << "Max nJet: " << max.nJet  << std::endl;
    std::cout << "Max nMuon: " << max.nMuon << std::endl;
    std::cout << "Max nSV: " << max.nSV   << std::endl;



    /**
     * @brief Calculates GenMET_pt mean and standard deviation.
     */
    auto [GenMET_mean, GenMET_std] = computeMeanAndStdGenMET(chain.get(), n_events, branches.GenMET_pt);

    std::cout << "Mean GenMET_pt: " << GenMET_mean << std::endl;
    std::cout << "Standard deviation GenMET_pt: " << GenMET_std << std::endl;



    /**
     * @brief Defines variables for Branches with variable arrays:
     * - unpacks static arrays;
     * - sets the Branch status to 1;
     * - associates the address of the Branch.
     */
    
    // Variables
    std::vector<Float_t> Jet_eta(max.nJet);
    std::vector<Float_t> Jet_pt(max.nJet);
    std::vector<Float_t> Jet_phi(max.nJet);
    std::vector<Float_t> Jet_mass(max.nJet);
    std::vector<Float_t> Jet_area(max.nJet);
    std::vector<Float_t> Jet_btagDeepFlavB(max.nJet);
    std::vector<Float_t> Jet_rawFactor(max.nJet);
    std::vector<Float_t> Jet_chHEF(max.nJet);
    std::vector<Float_t> Jet_neHEF(max.nJet);
    std::vector<Float_t> Jet_chEmEF(max.nJet);
    std::vector<Float_t> Jet_neEmEF(max.nJet);
    std::vector<Float_t> Jet_muEF(max.nJet);

    std::vector<Int_t> Muon_charge(max.nMuon);
    std::vector<Float_t> Muon_dxy(max.nMuon);
    std::vector<Float_t> Muon_dz(max.nMuon);
    std::vector<Float_t> Muon_eta(max.nMuon);
    std::vector<Float_t> Muon_mass(max.nMuon);
    std::vector<Float_t> Muon_phi(max.nMuon);
    std::vector<Float_t> Muon_pt(max.nMuon);

    std::vector<Float_t> SV_dlenSig(max.nSV);
    std::vector<Float_t> SV_mass(max.nSV);
    std::vector<Float_t> SV_eta(max.nSV);
    std::vector<Float_t> SV_phi(max.nSV);

    // Variables status
    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("Jet_area", 1);
    chain->SetBranchStatus("Jet_btagDeepFlavB", 1);
    chain->SetBranchStatus("Jet_rawFactor", 1);
    chain->SetBranchStatus("Jet_chHEF", 1);
    chain->SetBranchStatus("Jet_neHEF", 1);
    chain->SetBranchStatus("Jet_chEmEF", 1);
    chain->SetBranchStatus("Jet_neEmEF", 1);
    chain->SetBranchStatus("Jet_muEF", 1);

    chain->SetBranchStatus("Muon_charge", 1);
    chain->SetBranchStatus("Muon_dxy", 1);
    chain->SetBranchStatus("Muon_dz", 1);
    chain->SetBranchStatus("Muon_eta", 1);
    chain->SetBranchStatus("Muon_mass", 1);
    chain->SetBranchStatus("Muon_phi", 1);
    chain->SetBranchStatus("Muon_pt", 1);

    chain->SetBranchStatus("SV_dlenSig", 1);
    chain->SetBranchStatus("SV_mass", 1);
    chain->SetBranchStatus("SV_eta", 1);
    chain->SetBranchStatus("SV_phi", 1);

    // Variables address
    chain->SetBranchAddress("Jet_eta", Jet_eta.data());
    chain->SetBranchAddress("Jet_pt", Jet_pt.data());
    chain->SetBranchAddress("Jet_phi", Jet_phi.data());
    chain->SetBranchAddress("Jet_mass", Jet_mass.data());
    chain->SetBranchAddress("Jet_area", Jet_area.data());
    chain->SetBranchAddress("Jet_btagDeepFlavB", Jet_btagDeepFlavB.data());
    chain->SetBranchAddress("Jet_rawFactor", Jet_rawFactor.data());
    chain->SetBranchAddress("Jet_chHEF", Jet_chHEF.data());
    chain->SetBranchAddress("Jet_neHEF", Jet_neHEF.data());
    chain->SetBranchAddress("Jet_chEmEF", Jet_chEmEF.data());
    chain->SetBranchAddress("Jet_neEmEF", Jet_neEmEF.data());
    chain->SetBranchAddress("Jet_muEF", Jet_muEF.data());

    chain->SetBranchAddress("Muon_charge", Muon_charge.data());
    chain->SetBranchAddress("Muon_dxy", Muon_dxy.data());
    chain->SetBranchAddress("Muon_dz", Muon_dz.data());
    chain->SetBranchAddress("Muon_eta", Muon_eta.data());
    chain->SetBranchAddress("Muon_mass", Muon_mass.data());
    chain->SetBranchAddress("Muon_phi", Muon_phi.data());
    chain->SetBranchAddress("Muon_pt", Muon_pt.data());

    chain->SetBranchAddress("SV_dlenSig", SV_dlenSig.data());
    chain->SetBranchAddress("SV_mass", SV_mass.data());
    chain->SetBranchAddress("SV_eta", SV_eta.data());
    chain->SetBranchAddress("SV_phi", SV_phi.data());



    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);



    /**
     * @brief Defines output branches:
     * - 2 b-jets, 2 muons;
     * - best SV;
     * - engineered features.
     */
    // Best b-jet
    Float_t Jet_eta_bst, Jet_pt_bst, Jet_phi_bst, Jet_mass_bst;
    Float_t Jet_area_bst, Jet_btag_bst, Jet_rawFactor_bst;
    Float_t Jet_chHEF_bst, Jet_neHEF_bst, Jet_chEmEF_bst, Jet_neEmEF_bst, Jet_muEF_bst;

    // Second best b-jet
    Float_t Jet_eta_bnd, Jet_pt_bnd, Jet_phi_bnd, Jet_mass_bnd;
    Float_t Jet_area_bnd, Jet_btag_bnd, Jet_rawFactor_bnd;
    Float_t Jet_chHEF_bnd, Jet_neHEF_bnd, Jet_chEmEF_bnd, Jet_neEmEF_bnd, Jet_muEF_bnd;

    // First muon
    Int_t   Muon_charge_st;
    Float_t Muon_dxy_st, Muon_dz_st, Muon_eta_st;
    Float_t Muon_mass_st, Muon_phi_st, Muon_pt_st;

    // Second muon
    Int_t   Muon_charge_nd;
    Float_t Muon_dxy_nd, Muon_dz_nd, Muon_eta_nd;
    Float_t Muon_mass_nd, Muon_phi_nd, Muon_pt_nd;

    // Best SV (highest dlenSig)
    Float_t SV_dlenSig_bst, SV_mass_bst;
    
    // Engineered features
    Float_t M_mumu;  // invariant mass of the mu+mu- pair
    Float_t M_bb;  // invariant mass of the bb pair
    Float_t M_mumu_bb;  // invariant mass of the full mumu+bb system
    Float_t dR_MET_bb;  // deltaR between MET direction and bb system
    Float_t MET_projection_par;  // MET projected parallel to bb thrust axis
    Float_t MET_projection_perp;  // MET projected perpendicular to bb thrust axis
    Float_t dPhi_MET_mu1;  // delta phi between MET and leading muon
    Float_t dPhi_MET_jet1;  // delta phi between MET and leading b-jet
    Float_t HT;  // scalar sum of all jet pt

    // Best Jet
    newtree->Branch("Jet_eta_bst", &Jet_eta_bst);
    newtree->Branch("Jet_pt_bst", &Jet_pt_bst);
    newtree->Branch("Jet_phi_bst", &Jet_phi_bst);
    newtree->Branch("Jet_mass_bst", &Jet_mass_bst);
    newtree->Branch("Jet_area_bst", &Jet_area_bst);
    newtree->Branch("Jet_btag_bst", &Jet_btag_bst);
    newtree->Branch("Jet_rawFactor_bst", &Jet_rawFactor_bst);
    newtree->Branch("Jet_chHEF_bst", &Jet_chHEF_bst);
    newtree->Branch("Jet_neHEF_bst", &Jet_neHEF_bst);
    newtree->Branch("Jet_chEmEF_bst", &Jet_chEmEF_bst);
    newtree->Branch("Jet_neEmEF_bst", &Jet_neEmEF_bst);
    newtree->Branch("Jet_muEF_bst", &Jet_muEF_bst);

    // Second best Jet
    newtree->Branch("Jet_eta_bnd", &Jet_eta_bnd);
    newtree->Branch("Jet_pt_bnd", &Jet_pt_bnd);
    newtree->Branch("Jet_phi_bnd", &Jet_phi_bnd);
    newtree->Branch("Jet_mass_bnd", &Jet_mass_bnd);
    newtree->Branch("Jet_area_bnd", &Jet_area_bnd);
    newtree->Branch("Jet_btag_bnd", &Jet_btag_bnd);
    newtree->Branch("Jet_rawFactor_bnd",&Jet_rawFactor_bnd);
    newtree->Branch("Jet_chHEF_bnd", &Jet_chHEF_bnd);
    newtree->Branch("Jet_neHEF_bnd", &Jet_neHEF_bnd);
    newtree->Branch("Jet_chEmEF_bnd", &Jet_chEmEF_bnd);
    newtree->Branch("Jet_neEmEF_bnd", &Jet_neEmEF_bnd);
    newtree->Branch("Jet_muEF_bnd", &Jet_muEF_bnd);

    // First Muon
    newtree->Branch("Muon_charge_st", &Muon_charge_st);
    newtree->Branch("Muon_dxy_st", &Muon_dxy_st);
    newtree->Branch("Muon_dz_st", &Muon_dz_st);
    newtree->Branch("Muon_eta_st", &Muon_eta_st);
    newtree->Branch("Muon_mass_st", &Muon_mass_st);
    newtree->Branch("Muon_phi_st", &Muon_phi_st);
    newtree->Branch("Muon_pt_st", &Muon_pt_st);

    // Second Muon
    newtree->Branch("Muon_charge_nd", &Muon_charge_nd);
    newtree->Branch("Muon_dxy_nd", &Muon_dxy_nd);
    newtree->Branch("Muon_dz_nd", &Muon_dz_nd);
    newtree->Branch("Muon_eta_nd", &Muon_eta_nd);
    newtree->Branch("Muon_mass_nd", &Muon_mass_nd);
    newtree->Branch("Muon_phi_nd", &Muon_phi_nd);
    newtree->Branch("Muon_pt_nd", &Muon_pt_nd);

    // Best SV
    newtree->Branch("SV_dlenSig_bst",&SV_dlenSig_bst);
    newtree->Branch("SV_mass_bst", &SV_mass_bst);

    // Engineered features
    newtree->Branch("M_mumu", &M_mumu);
    newtree->Branch("M_bb", &M_bb);
    newtree->Branch("M_mumu_bb", &M_mumu_bb);
    newtree->Branch("dR_MET_bb", &dR_MET_bb);
    newtree->Branch("MET_projection_par", &MET_projection_par);
    newtree->Branch("MET_projection_perp",&MET_projection_perp);
    newtree->Branch("dPhi_MET_mu1", &dPhi_MET_mu1);
    newtree->Branch("dPhi_MET_jet1",&dPhi_MET_jet1);
    newtree->Branch("HT", &HT);



    /**
     * @brief Selects only the events of interest.
     * Implements sanity checks over the physics of many features, orginal or extracted.
     * 
     * Keeps only events with at least 2 muons and 2 b-jets,
     * and applies a threshold for the b-jet quality, later defining new branches accordingly.
     * 
     * Defines new useful variables through feature engineering.
     */
    Long64_t n_events_remaining = 0;
    const Float_t btag_threshold = 0.7f;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);

        if (branches.MET_pt <= 0) continue;
        if (branches.nMuon < 2) continue;
        if (branches.nJet < 2) continue;
        
        auto [maxIdx, secondMaxIdx] = findTwoMaxIndices(Jet_btagDeepFlavB, branches.nJet);

        // First two sorted b-tags
        Jet_btag_bst = Jet_btagDeepFlavB[maxIdx];
        Jet_btag_bnd = Jet_btagDeepFlavB[secondMaxIdx];

        if (Jet_btag_bst <= btag_threshold || Jet_btag_bnd <= btag_threshold) continue;

        // Jets
        Jet_eta_bst = Jet_eta[maxIdx];
        Jet_pt_bst = Jet_pt[maxIdx];
        Jet_phi_bst = Jet_phi[maxIdx];
        Jet_mass_bst = Jet_mass[maxIdx];
        Jet_area_bst = Jet_area[maxIdx];
        Jet_rawFactor_bst= Jet_rawFactor[maxIdx];
        Jet_chHEF_bst = Jet_chHEF[maxIdx];
        Jet_neHEF_bst = Jet_neHEF[maxIdx];
        Jet_chEmEF_bst = Jet_chEmEF[maxIdx];
        Jet_neEmEF_bst = Jet_neEmEF[maxIdx];
        Jet_muEF_bst = Jet_muEF[maxIdx];

        Jet_eta_bnd = Jet_eta[secondMaxIdx];
        Jet_pt_bnd = Jet_pt[secondMaxIdx];
        Jet_phi_bnd = Jet_phi[secondMaxIdx];
        Jet_mass_bnd = Jet_mass[secondMaxIdx];
        Jet_area_bnd = Jet_area[secondMaxIdx];
        Jet_rawFactor_bnd= Jet_rawFactor[secondMaxIdx];
        Jet_chHEF_bnd = Jet_chHEF[secondMaxIdx];
        Jet_neHEF_bnd = Jet_neHEF[secondMaxIdx];
        Jet_chEmEF_bnd = Jet_chEmEF[secondMaxIdx];
        Jet_neEmEF_bnd = Jet_neEmEF[secondMaxIdx];
        Jet_muEF_bnd = Jet_muEF[secondMaxIdx];

        if (Jet_pt_bst <= 0) continue;
        if (Jet_pt_bnd <= 0) continue;

        // Muons
        Muon_charge_st = Muon_charge[0];
        Muon_dxy_st = Muon_dxy[0];
        Muon_dz_st = Muon_dz[0];
        Muon_eta_st = Muon_eta[0];
        Muon_mass_st = Muon_mass[0];
        Muon_phi_st = Muon_phi[0];
        Muon_pt_st = Muon_pt[0];
                
        Muon_charge_nd = Muon_charge[1];
        Muon_dxy_nd = Muon_dxy[1];
        Muon_dz_nd = Muon_dz[1];
        Muon_eta_nd = Muon_eta[1];
        Muon_mass_nd = Muon_mass[1];
        Muon_phi_nd = Muon_phi[1];
        Muon_pt_nd = Muon_pt[1];

        if (Muon_pt_st <= 0) continue;
        if (Muon_pt_nd <= 0) continue;

        // Best Secondary Vertex
        SV_dlenSig_bst = -1.0;
        SV_mass_bst = -1.0;
        if (branches.nSV > 0) {
            size_t bestSV = 0;
            for (UInt_t s = 1; s < branches.nSV; s++) {
                if (SV_dlenSig[s] > SV_dlenSig[bestSV]) bestSV = s;
            }
            SV_dlenSig_bst = SV_dlenSig[bestSV];
            SV_mass_bst    = SV_mass[bestSV];
        }

        // TLorentzVectors for the two muons and the two b-jets
        TLorentzVector lv_mu1 = buildLV(Muon_pt_st, Muon_eta_st, Muon_phi_st, Muon_mass_st);
        TLorentzVector lv_mu2 = buildLV(Muon_pt_nd, Muon_eta_nd, Muon_phi_nd, Muon_mass_nd);
        TLorentzVector lv_jet1 = buildLV(Jet_pt_bst, Jet_eta_bst, Jet_phi_bst, Jet_mass_bst);
        TLorentzVector lv_jet2 = buildLV(Jet_pt_bnd, Jet_eta_bnd, Jet_phi_bnd, Jet_mass_bnd);

        TLorentzVector lv_mumu = lv_mu1 + lv_mu2;
        TLorentzVector lv_bb = lv_jet1 + lv_jet2;
        TLorentzVector lv_mumu_bb= lv_mumu + lv_bb;

        // Invariant masses
        M_mumu = lv_mumu.M();
        M_bb = lv_bb.M();
        M_mumu_bb = lv_mumu_bb.M();

        if (M_mumu < 0.2) continue; 

        // DeltaR between MET direction and bb system
        // MET eta = 0 by convention
        dR_MET_bb = deltaR(0.0, branches.MET_phi, lv_bb.Eta(), lv_bb.Phi());

        // MET projections along and perpendicular to the bb axis
        Float_t dphi_MET_bb = deltaPhi(branches.MET_phi, lv_bb.Phi());
        MET_projection_par = branches.MET_pt * std::cos(dphi_MET_bb);
        MET_projection_perp = branches.MET_pt * std::sin(dphi_MET_bb);

        // Delta phi between MET and leading objects
        dPhi_MET_mu1 = deltaPhi(branches.MET_phi, Muon_phi_st);
        dPhi_MET_jet1 = deltaPhi(branches.MET_phi, Jet_phi_bst);

        // HT: scalar sum of pt of all jets in the event
        HT = 0.0;
        for (UInt_t j = 0; j < branches.nJet; j++) HT += Jet_pt[j];

        // Fill
        newtree->Fill();
        n_events_remaining++;
    }

    /**
     * @brief Number of events remaining after the skimming.
     */
    std::cout << "Remaining events after skimming: " << n_events_remaining << std::endl;

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../CleanedDatasets/cleaned_HToAATo2Mu2B.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}