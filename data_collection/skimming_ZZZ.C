#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void skimming_ZZZ() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("../datasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");



    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);



    /**
     * @brief Defines input variable.
     * Only selects entries that are interesting for the ML model.
     * The entries have name and type of the dataset Branches.
     */
    // Single value
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

    // Multiple values
    const int n_max_tracks = 100;

    UInt_t nSV;
    Int_t SV_charge[n_max_tracks];
    Float_t SV_chi2[n_max_tracks];
    Float_t SV_dxy[n_max_tracks];
    Float_t SV_pAngle[n_max_tracks];

    UInt_t nJet;
    Float_t Jet_area[n_max_tracks];
    Float_t Jet_eta[n_max_tracks];
    Float_t Jet_mass[n_max_tracks];
    Float_t Jet_phi[n_max_tracks];
    Float_t Jet_pt[n_max_tracks];
    Float_t Jet_btagDeepFlavB[n_max_tracks];



    /**
     * @brief Sets previous Branches status to 1.
     */
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
    chain->SetBranchStatus("SV_charge", 1);
    chain->SetBranchStatus("SV_chi2", 1);
    chain->SetBranchStatus("SV_dxy", 1);
    chain->SetBranchStatus("SV_pAngle", 1);
    chain->SetBranchStatus("nJet", 1);
    chain->SetBranchStatus("Jet_area", 1);
    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_btagDeepFlavB", 1);



    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     */
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
    chain->SetBranchAddress("SV_charge", &SV_charge);
    chain->SetBranchAddress("SV_chi2", &SV_chi2);
    chain->SetBranchAddress("SV_dxy", &SV_dxy);
    chain->SetBranchAddress("SV_pAngle", &SV_pAngle);
    chain->SetBranchAddress("nJet", &nJet);
    chain->SetBranchAddress("Jet_area", &Jet_area);
    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);



    /**
     * @param n_events Number of events in each file.
     */
    Long64_t n_events = chain->GetEntries();

    std::cout << "nEvents before skimming:" << n_events << std::endl;



    /**
     * @brief Gets the maximum and minimum number of jets.
     */
    UInt_t max_nJet = 0;
    UInt_t min_nJet = 0;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }

        if (min_nJet > nJet) {
            min_nJet = nJet;
        }
    }
    std::cout << "MAX number of Jets is: " << max_nJet << std::endl;
    std::cout << "MIN number of Jets is: " << min_nJet << std::endl;



    /**
     * @brief Calculate mean and standard deviation of GenMET
     * to later perform a statistical cut on outliers.
     */
    Float_t sum = 0.0;
    Float_t sum2 = 0.0;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        sum += GenMET_pt;
    }

    Float_t GenMET_mean = sum / n_events;
    Float_t GenMET_variance_num = 0.0;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        GenMET_variance_num += (GenMET_pt - GenMET_mean)*(GenMET_pt - GenMET_mean);
    }

    Float_t GenMET_variance = GenMET_variance_num / n_events;
    Float_t GenMET_std = std::sqrt(GenMET_variance);

    std::cout << "Mean GenMET_pt: " << GenMET_mean << std::endl;
    std::cout << "Standard deviation GenMET_pt: " << GenMET_std << std::endl;



    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);



    /**
     * @brief Takes the first two Jets and unpacks the first SV.
     */
    // Variable definition
    Int_t SV_charge_st;
    Float_t SV_chi2_st;
    Float_t SV_dxy_st;
    Float_t SV_pAngle_st;

    Float_t Jet_area_st,Jet_area_nd;
    Float_t Jet_eta_st, Jet_eta_nd;
    Float_t Jet_mass_st, Jet_mass_nd;
    Float_t Jet_phi_st, Jet_phi_nd;
    Float_t Jet_pt_st, Jet_pt_nd;
    Float_t Jet_btag_st, Jet_btag_nd;

    // Addresses
    newtree->Branch("SV_charge_st", &SV_charge_st);
    newtree->Branch("SV_chi2_st", &SV_chi2_st);
    newtree->Branch("SV_dxy_st", &SV_dxy_st);
    newtree->Branch("SV_pAngle_st", &SV_pAngle_st);

    newtree->Branch("Jet_area_st", &Jet_area_st);
    newtree->Branch("Jet_eta_st", &Jet_eta_st);
    newtree->Branch("Jet_mass_st", &Jet_mass_st);
    newtree->Branch("Jet_phi_st", &Jet_phi_st);
    newtree->Branch("Jet_pt_st", &Jet_pt_st);
    newtree->Branch("Jet_btag_st", &Jet_btag_st);

    newtree->Branch("Jet_area_nd", &Jet_area_nd);
    newtree->Branch("Jet_eta_nd", &Jet_eta_nd);
    newtree->Branch("Jet_mass_nd", &Jet_mass_nd);
    newtree->Branch("Jet_phi_nd", &Jet_phi_nd);
    newtree->Branch("Jet_pt_nd", &Jet_pt_nd);
    newtree->Branch("Jet_btag_nd", &Jet_btag_nd);

    Long64_t n_events_skimmed = 0;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);

        if (nSV > 0 && nJet > 1) {
            SV_charge_st = SV_charge[0];
            SV_chi2_st = SV_chi2[0];
            SV_dxy_st = SV_dxy[0];
            SV_pAngle_st = SV_pAngle[0];

            Jet_area_st = Jet_area[0];
            Jet_eta_st = Jet_eta[0];
            Jet_mass_st = Jet_mass[0];
            Jet_phi_st = Jet_phi[0];
            Jet_pt_st = Jet_pt[0];
            Jet_btag_st = Jet_btagDeepFlavB[0];

            Jet_area_nd = Jet_area[1];
            Jet_eta_nd = Jet_eta[1];
            Jet_mass_nd = Jet_mass[1];
            Jet_phi_nd = Jet_phi[1];
            Jet_pt_nd = Jet_pt[1];
            Jet_btag_nd = Jet_btagDeepFlavB[1];

        newtree->Fill();
        n_events_skimmed++;
        }
    }

    /**
     * @brief Number of events remaining after the skimming.
     */
    std::cout << "Remaining events after skimming: " << n_events_skimmed << std::endl;

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/specific_skimmed_ZZZ.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}