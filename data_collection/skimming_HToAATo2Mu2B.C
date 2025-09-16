#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <utility>
#include <cstddef>

std::pair<size_t, size_t> findTwoMaxIndices(const Float_t* arr, size_t n) {
    if (n < 2) {
        throw std::invalid_argument("L'array deve contenere almeno 2 elementi");
    }

    size_t maxIdx = 0;
    size_t secondMaxIdx = 1;

    // inizializzazione: assicuriamoci che maxIdx punti al massimo iniziale
    if (arr[secondMaxIdx] > arr[maxIdx]) {
        std::swap(maxIdx, secondMaxIdx);
    }

    for (size_t i = 2; i < n; ++i) {
        if (arr[i] > arr[maxIdx]) {
            secondMaxIdx = maxIdx;
            maxIdx = i;
        } else if (arr[i] > arr[secondMaxIdx] && i != maxIdx) {
            secondMaxIdx = i;
        }
    }

    return {maxIdx, secondMaxIdx};
}

void skimming_HToAATo2Mu2B() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("../datasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");



    /**
     * @param n_events Number of events in each file.
     */
    Long64_t n_events = chain->GetEntries();

    std::cout << "nEvents before skimming:" << n_events << std::endl;



    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);



    /**
     * @brief Defines single value input global variables:
     * - the entries have name and type of the dataset Branches;
     * - the entries Branch status is set to one;
     * - gets the address of the Branches in order to copy their values.
     * TODO: add check if Branches don't exist in the original dataset.
     */
    // Single value variables
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

    std::cout << "Defined single value variables." << std::endl;

    // Single variables status
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

    std::cout << "Set Branch status to 1." << std::endl;

    // Single variables address
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

    std::cout << "Set all Branches addresses." << std::endl;



    /**
     * @brief Gets the maximum number of jets.
     */
    UInt_t max_nJet = 0;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }
    }

    std::cout << "Max number of JETS is: " << max_nJet << std::endl;



    /**
     * @brief Gets the maximum number of jets.
     */
    UInt_t max_nMuon = 0;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (nMuon > max_nMuon) {
            max_nMuon = nMuon;
        }
    }

    std::cout << "Max number of MUONS is: " << max_nMuon << std::endl;



    /**
     * @brief Defines variables for Branches with variable arrays:
     * - unpacks static arrays;
     * - sets the Branch status to 1;
     * - associates the address of the Branch.
     */
    const UInt_t const_max_nJet = max_nJet;
    std::cout << "Const max number of JETS is: " << const_max_nJet << std::endl;

    const UInt_t const_max_nMuon = max_nMuon;
    std::cout << "Const Max number of MUONS is: " << const_max_nMuon << std::endl;

    // Variables
    Float_t Jet_eta[const_max_nJet];
    Float_t Jet_pt[const_max_nJet];
    Float_t Jet_phi[const_max_nJet];
    Float_t Jet_mass[const_max_nJet];
    Float_t Jet_area[const_max_nJet];
    Float_t Jet_btagDeepFlavB[const_max_nJet];

    Int_t Muon_charge[const_max_nMuon];
    Float_t Muon_dxy[const_max_nMuon];
    Float_t Muon_dz[const_max_nMuon];
    Float_t Muon_eta[const_max_nMuon];
    Float_t Muon_mass[const_max_nMuon];
    Float_t Muon_phi[const_max_nMuon];
    Float_t Muon_pt[const_max_nMuon];

    // Variables status
    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("Jet_area", 1);
    chain->SetBranchStatus("Jet_btagDeepFlavB", 1);

    chain->SetBranchStatus("Muon_charge", 1);
    chain->SetBranchStatus("Muon_dxy", 1);
    chain->SetBranchStatus("Muon_dz", 1);
    chain->SetBranchStatus("Muon_eta", 1);
    chain->SetBranchStatus("Muon_mass", 1);
    chain->SetBranchStatus("Muon_phi", 1);
    chain->SetBranchStatus("Muon_pt", 1);

    // Variables address
    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    chain->SetBranchAddress("Jet_area", &Jet_area);
    chain->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);

    chain->SetBranchAddress("Muon_charge", &Muon_charge);
    chain->SetBranchAddress("Muon_dxy", &Muon_dxy);
    chain->SetBranchAddress("Muon_dz", &Muon_dz);
    chain->SetBranchAddress("Muon_eta", &Muon_eta);
    chain->SetBranchAddress("Muon_mass", &Muon_mass);
    chain->SetBranchAddress("Muon_phi", &Muon_phi);
    chain->SetBranchAddress("Muon_pt", &Muon_pt);



    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);



    /**
     * @brief Calculate mean and standard deviation of GenMET.
     */
    Float_t sum = 0.0;
    Float_t sum2 = 0.0;

    for (Long64_t i = 0; i < n_events; i++) {
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
     * @brief Selects only the first two jets above b-tag threshold
     * and defines them as new branches.
     * Gets only the frst two muons.
     */
    Float_t Jet_eta_bst, Jet_eta_bnd;
    Float_t Jet_pt_bst, Jet_pt_bnd;
    Float_t Jet_phi_bst, Jet_phi_bnd;
    Float_t Jet_mass_bst, Jet_mass_bnd;
    Float_t Jet_area_bst, Jet_area_bnd;
    Float_t Jet_btag_bst, Jet_btag_bnd;

    Int_t Muon_charge_st, Muon_charge_nd;
    Float_t Muon_dxy_st, Muon_dxy_nd;
    Float_t Muon_dz_st, Muon_dz_nd;
    Float_t Muon_eta_st, Muon_eta_nd;
    Float_t Muon_mass_st, Muon_mass_nd;
    Float_t Muon_phi_st, Muon_phi_nd;
    Float_t Muon_pt_st, Muon_pt_nd;

    // Best Jet
    newtree->Branch("Jet_eta_bst", &Jet_eta_bst);
    newtree->Branch("Jet_pt_bst", &Jet_pt_bst);
    newtree->Branch("Jet_phi_bst", &Jet_phi_bst);
    newtree->Branch("Jet_mass_bst", &Jet_mass_bst);
    newtree->Branch("Jet_area_bst", &Jet_area_bst);
    newtree->Branch("Jet_btag_bst", &Jet_btag_bst);

    // Second best Jet
    newtree->Branch("Jet_eta_bnd", &Jet_eta_bnd);
    newtree->Branch("Jet_pt_bnd", &Jet_pt_bnd);
    newtree->Branch("Jet_phi_bnd", &Jet_phi_bnd);
    newtree->Branch("Jet_mass_bnd", &Jet_mass_bnd);
    newtree->Branch("Jet_area_bnd", &Jet_area_bnd);
    newtree->Branch("Jet_btag_bnd", &Jet_btag_bnd);

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



    Long64_t n_events_skimmed = 0;
    Float_t btag_threshold = 0.7;

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);

        if (nMuon==2 && nJet>1) {
            auto [maxIdx, secondMaxIdx] = findTwoMaxIndices(Jet_btagDeepFlavB, max_nJet);

            // First two sorted b-tags
            Jet_btag_bst = Jet_btagDeepFlavB[maxIdx];
            Jet_btag_bnd = Jet_btagDeepFlavB[secondMaxIdx];

            if (Jet_btag_bst>btag_threshold && Jet_btag_bnd>btag_threshold) {
                // Best Jet
                Jet_eta_bst = Jet_eta[maxIdx];
                Jet_pt_bst = Jet_pt[maxIdx];
                Jet_phi_bst = Jet_phi[maxIdx];
                Jet_mass_bst = Jet_mass[maxIdx];
                Jet_area_bst = Jet_area[maxIdx];

                // Second best Jet
                Jet_eta_bnd = Jet_eta[secondMaxIdx];
                Jet_pt_bnd = Jet_pt[secondMaxIdx];
                Jet_phi_bnd = Jet_phi[secondMaxIdx];
                Jet_mass_bnd = Jet_mass[secondMaxIdx];
                Jet_area_bnd = Jet_area[secondMaxIdx];

                // First Muon
                Muon_charge_st = Muon_charge[0];
                Muon_dxy_st = Muon_dxy[0];
                Muon_dz_st = Muon_dz[0];
                Muon_eta_st = Muon_eta[0];
                Muon_mass_st = Muon_mass[0];
                Muon_phi_st = Muon_phi[0];
                Muon_pt_st = Muon_pt[0];
                
                // Second Muon
                Muon_charge_nd = Muon_charge[1];
                Muon_dxy_nd = Muon_dxy[1];
                Muon_dz_nd = Muon_dz[1];
                Muon_eta_nd = Muon_eta[1];
                Muon_mass_nd = Muon_mass[1];
                Muon_phi_nd = Muon_phi[1];
                Muon_pt_nd = Muon_pt[1];

                newtree->Fill();
                n_events_skimmed++;
            }
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
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/specific_skimmed_HToAATo2Mu2B.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}