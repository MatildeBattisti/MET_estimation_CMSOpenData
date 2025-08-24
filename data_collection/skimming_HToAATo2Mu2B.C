#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void skimming_HToAATo2Mu2B() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    //chain->Add("../datasets/6357E7BC-502C-2E45-A649-73A57B651715.root");  // dataset 0
    chain->Add("../datasets/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root");  // dataset 1
    //chain->Add("../datasets/77DB0F5B-4123-4E4B-A9D0-3CEBA8575834.root");  // dataset 2
    //chain->Add("../datasets/048A040C-DA63-1949-9BA7-075371EB4296.root");  // dataset 3
   
    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);

    /**
     * @brief Only selects entries interesting for the ML model.
     * The entries have name and type of the original ones.
     */
    UInt_t nJet;

    Float_t MET_pt;
    Float_t MET_phi;
    Float_t MET_significance;
    Float_t MET_covXX;
    Float_t MET_covXY;
    Float_t MET_covYY;
    Float_t GenMET_pt;
    Float_t CaloMET_pt;
    Float_t CaloMET_phi;

    Float_t PV_x;
    Float_t PV_y;
    Float_t PV_z;
    Float_t PV_chi2;

    const int maxNJets = 19;

    Float_t Jet_eta[maxNJets];
    Float_t Jet_pt[maxNJets];
    Float_t Jet_phi[maxNJets];
    Float_t Jet_mass[maxNJets];
    Float_t Jet_area[maxNJets];
    Float_t Jet_btagDeepFlavB[maxNJets];

    UInt_t nMuon;

    Float_t Muon_eta[maxNJets];
    Float_t Muon_pt[maxNJets];
    Float_t Muon_phi[maxNJets];
    Float_t Muon_mass[maxNJets];
    Int_t Muon_charge[maxNJets];

    /**
     * @brief Selects the previous branches, setting their
     * status to one.
     */
    chain->SetBranchStatus("nJet", 1);

    chain->SetBranchStatus("MET_pt", 1);
    chain->SetBranchStatus("MET_phi", 1);
    chain->SetBranchStatus("MET_significance", 1);
    chain->SetBranchStatus("MET_covXX", 1);
    chain->SetBranchStatus("MET_covXY", 1);
    chain->SetBranchStatus("MET_covYY", 1);
    chain->SetBranchStatus("GenMET_pt", 1);
    chain->SetBranchStatus("CaloMET_pt", 1);
    chain->SetBranchStatus("CaloMET_phi", 1);

    chain->SetBranchStatus("PV_x", 1);
    chain->SetBranchStatus("PV_y", 1);
    chain->SetBranchStatus("PV_z", 1);
    chain->SetBranchStatus("PV_chi2", 1);

    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("Jet_area", 1);
    chain->SetBranchStatus("Jet_btagDeepFlavB", 1);

    chain->SetBranchStatus("nMuon", 1);

    chain->SetBranchStatus("Muon_eta", 1);
    chain->SetBranchStatus("Muon_pt", 1);
    chain->SetBranchStatus("Muon_phi", 1);
    chain->SetBranchStatus("Muon_mass", 1);
    chain->SetBranchStatus("Muon_charge", 1);
    
    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     **/
    chain->SetBranchAddress("nJet", &nJet);

    chain->SetBranchAddress("MET_pt", &MET_pt);
    chain->SetBranchAddress("MET_phi", &MET_phi);
    chain->SetBranchAddress("MET_significance", &MET_significance);
    chain->SetBranchAddress("MET_covXX", &MET_covXX);
    chain->SetBranchAddress("MET_covXY", &MET_covXY);
    chain->SetBranchAddress("MET_covYY", &MET_covYY);
    chain->SetBranchAddress("GenMET_pt", &GenMET_pt);
    chain->SetBranchAddress("CaloMET_pt", &CaloMET_pt);
    chain->SetBranchAddress("CaloMET_phi", &CaloMET_phi);

    chain->SetBranchAddress("PV_x", &PV_x);
    chain->SetBranchAddress("PV_y", &PV_y);
    chain->SetBranchAddress("PV_z", &PV_z);
    chain->SetBranchAddress("PV_chi2", &PV_chi2);

    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    chain->SetBranchAddress("Jet_area", &Jet_area);
    chain->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);

    chain->SetBranchAddress("nMuon", &nMuon);

    chain->SetBranchAddress("Muon_eta", &Muon_eta);
    chain->SetBranchAddress("Muon_pt", &Muon_pt);
    chain->SetBranchAddress("Muon_phi", &Muon_phi);
    chain->SetBranchAddress("Muon_mass", &Muon_mass);
    chain->SetBranchAddress("Muon_charge", &Muon_charge);

    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);
    
    /**
     * @param n_events Number of events in each file
     */
    Long64_t n_events = chain->GetEntries();

    /**
     * @brief Gets the maximum number of jets.
     */
    Float_t max_nJet;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }
    }

    std::cout << "Max number of Jets is: " << max_nJet << std::endl;

    /**
     * @brief Calculate mean and standard deviation of GenMET
     * to later perform a statistical cut on outliers.
     */
    Float_t sum = 0.0;
    Float_t sum2 = 0.0;
    Long64_t count = 0;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        sum += GenMET_pt;
        count++;
    }

    Float_t GenMET_mean = sum / count;
    Float_t GenMET_variance_num = 0.0;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        GenMET_variance_num += (GenMET_pt - GenMET_mean)*(GenMET_pt - GenMET_mean);
    }

    Float_t GenMET_variance = GenMET_variance_num / count;
    Float_t GenMET_std = std::sqrt(GenMET_variance);

    Float_t GenMET_cut = GenMET_mean + 10*GenMET_std;

    std::cout << "Mean GenMET_pt: " << GenMET_mean << std::endl;
    std::cout << "Standard deviation GenMET_pt: " << GenMET_std << std::endl;
    std::cout << "GenMET_pt cut performed at: " << GenMET_cut << std::endl;

    /**
     * @brief Selects only the first two jets above b-tag threshold
     * and defines them as new branches.
     */
    Float_t Jet_eta_bst, Jet_eta_bnd;
    Float_t Jet_pt_bst, Jet_pt_bnd;
    Float_t Jet_phi_bst, Jet_phi_bnd;
    Float_t Jet_mass_bst, Jet_mass_bnd;
    Float_t Jet_area_bst, Jet_area_bnd;
    Float_t Jet_btag_bst, Jet_btag_bnd;

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

    Float_t Muon_eta_st, Muon_eta_nd;
    Float_t Muon_pt_st, Muon_pt_nd;
    Float_t Muon_phi_st, Muon_phi_nd;
    Float_t Muon_mass_st, Muon_mass_nd;
    Float_t Muon_charge_st, Muon_charge_nd;

    // First Muon
    newtree->Branch("Muon_eta_st", &Muon_eta_st);
    newtree->Branch("Muon_pt_st", &Muon_pt_st);
    newtree->Branch("Muon_phi_st", &Muon_phi_st);
    newtree->Branch("Muon_mass_st", &Muon_mass_st);
    newtree->Branch("Muon_charge_st", &Muon_charge_st);
    
    // Second Muon
    newtree->Branch("Muon_eta_nd", &Muon_eta_nd);
    newtree->Branch("Muon_pt_nd", &Muon_pt_nd);
    newtree->Branch("Muon_phi_nd", &Muon_phi_nd);
    newtree->Branch("Muon_mass_nd", &Muon_mass_nd);
    newtree->Branch("Muon_charge_nd", &Muon_charge_nd);

    Float_t Muon_E_st, Muon_E_nd;
    Float_t Muon_Deltaeta;
    Float_t Muon_Deltaphi;
    Float_t Muon_DeltaR;
    Float_t Muon_InvMass;
    Float_t Deltaphi_METJbest, Deltaphi_METJbnd;
    Float_t Deltaphi_METmst, Deltaphi_METmnd;
    Float_t pt_sum;

    // Derived quantities
    newtree->Branch("Muon_Deltaeta", &Muon_Deltaeta);
    newtree->Branch("Muon_Deltaphi", &Muon_Deltaphi);
    newtree->Branch("Muon_DeltaR", &Muon_DeltaR);
    newtree->Branch("Muon_InvMass", &Muon_InvMass);
    newtree->Branch("Deltaphi_METJbest", &Deltaphi_METJbest);
    newtree->Branch("Deltaphi_METJbnd", &Deltaphi_METJbnd);
    newtree->Branch("Deltaphi_METmst", &Deltaphi_METmst);
    newtree->Branch("Deltaphi_METmnd", &Deltaphi_METmnd);
    newtree->Branch("pt_sum", &pt_sum);

    Float_t btag_threshold = 0.7;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);

        // Create array of indices
        int indices[maxNJets];
        for (int i = 0; i < maxNJets; ++i)
            indices[i] = i;

        // Sort indices based on values on btag descending
        std::sort(indices, indices + maxNJets, [&](int i, int j) {
            return Jet_btagDeepFlavB[i] > Jet_btagDeepFlavB[j];
        });

        // Create sorted arrays
        Float_t Jet_btagDeepFlavB_sorted[maxNJets];
        Float_t Jet_eta_sorted[maxNJets];
        Float_t Jet_pt_sorted[maxNJets];
        Float_t Jet_phi_sorted[maxNJets];
        Float_t Jet_mass_sorted[maxNJets];
        Float_t Jet_area_sorted[maxNJets];

        for (int i = 0; i < maxNJets; ++i) {
            Jet_btagDeepFlavB_sorted[i] = Jet_btagDeepFlavB[indices[i]];
            Jet_eta_sorted[i] = Jet_eta[indices[i]];
            Jet_pt_sorted[i] = Jet_pt[indices[i]];
            Jet_phi_sorted[i] = Jet_phi[indices[i]];
            Jet_mass_sorted[i] = Jet_mass[indices[i]];
            Jet_area_sorted[i] = Jet_area[indices[i]];
        }

        // First two sorted b-tags
        Jet_btag_bst = (nJet > 0) ? Jet_btagDeepFlavB_sorted[0] : 0.0f;
        Jet_btag_bnd = (nJet > 1) ? Jet_btagDeepFlavB_sorted[1] : 0.0f;

        if (Jet_btag_bst>btag_threshold && Jet_btag_bnd>btag_threshold && GenMET_pt<GenMET_cut) {
            // Best Jet
            Jet_eta_bst = (nJet > 0) ? Jet_eta_sorted[0] : 0.0f;
            Jet_pt_bst = (nJet > 0) ? Jet_pt_sorted[0] : 0.0f;
            Jet_phi_bst = (nJet > 0) ? Jet_phi_sorted[0] : 0.0f;
            Jet_mass_bst = (nJet > 0) ? Jet_mass_sorted[0] : 0.0f;
            Jet_area_bst = (nJet > 0) ? Jet_area_sorted[0] : 0.0f;

            // Second best Jet
            Jet_eta_bnd = (nJet > 1) ? Jet_eta_sorted[1] : 0.0f;
            Jet_pt_bnd = (nJet > 1) ? Jet_pt_sorted[1] : 0.0f;
            Jet_phi_bnd = (nJet > 1) ? Jet_phi_sorted[1] : 0.0f;
            Jet_mass_bnd = (nJet > 1) ? Jet_mass_sorted[1] : 0.0f;
            Jet_area_bnd = (nJet > 1) ? Jet_area_sorted[1] : 0.0f;

            // First Muon
            Muon_eta_st = (nMuon > 0) ? Muon_eta[0] : 0.0f;
            Muon_pt_st = (nMuon > 0) ? Muon_pt[0] : 0.0f;
            Muon_phi_st = (nMuon > 0) ? Muon_phi[0] : 0.0f;
            Muon_mass_st = (nMuon > 0) ? Muon_mass[0] : 0.0f;
            Muon_charge_st = (nMuon > 0) ? Muon_charge[0] : 0.0f;

            // Second Muon
            Muon_eta_nd = (nMuon > 1) ? Muon_eta[1] : 0.0f;
            Muon_pt_nd = (nMuon > 1) ? Muon_pt[1] : 0.0f;
            Muon_phi_nd = (nMuon > 1) ? Muon_phi[1] : 0.0f;
            Muon_mass_nd = (nMuon > 1) ? Muon_mass[1] : 0.0f;
            Muon_charge_nd = (nMuon > 1) ? Muon_charge[1] : 0.0f;

            // Derived quantities
            Muon_Deltaeta = Muon_eta_st - Muon_eta_nd;
            Muon_Deltaphi = Muon_phi_st - Muon_phi_nd;
            Muon_DeltaR = std::sqrt(Muon_Deltaeta*Muon_Deltaeta + Muon_Deltaphi*Muon_Deltaphi);
            
            Muon_E_st = std::sqrt(Muon_pt_st*Muon_pt_st*std::cosh(Muon_eta_st)*std::cosh(Muon_eta_st)+Muon_mass_st*Muon_mass_st);
            Muon_E_nd = std::sqrt(Muon_pt_nd*Muon_pt_nd*std::cosh(Muon_eta_nd)*std::cosh(Muon_eta_nd)+Muon_mass_nd*Muon_mass_nd);
        
            Muon_InvMass = std::sqrt((Muon_E_st+Muon_E_nd)*(Muon_E_st+Muon_E_nd)-
                std::abs(Muon_pt_st+Muon_pt_nd)*std::abs(Muon_pt_st+Muon_pt_nd));

            Deltaphi_METJbest = MET_phi - Jet_phi_bst;
            Deltaphi_METJbnd = MET_phi - Jet_phi_bnd;
            Deltaphi_METmst = MET_phi - Muon_phi_st;
            Deltaphi_METmnd = MET_phi - Muon_phi_nd;

            pt_sum = Jet_pt_bst + Jet_pt_bnd + Muon_pt_st + Muon_pt_nd;
        
        newtree->Fill();
        }
    }

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/skimmed1_specific.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}
