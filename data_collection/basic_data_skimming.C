#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void basic_data_skimming() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    //chain->Add("../datasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");  // dataset 0
    //chain->Add("../datasets/HToAATo2Mu2B/86EE006C-6402-E94B-9BC2-05860851DEE3.root");  // dataset 1
    //chain->Add("../datasets/HToAATo2Mu2B/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root");  // dataset 2
    //chain->Add("../datasets/HToAATo2Mu2B/048A040C-DA63-1949-9BA7-075371EB4296.root");  // dataset 3
   
    //chain->Add("../datasets/ZZZ/2E96A5E9-C938-A149-BBBF-8FD81A9E5AD6.root");
    //chain->Add("../datasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");

    chain->Add("../datasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");

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
    Float_t MET_covXX;
    Float_t MET_covXY;
    Float_t MET_covYY;
    Float_t MET_significance;
    Float_t GenMET_pt;

    const int maxNJets = 25;

    Float_t Jet_area[maxNJets];
    Float_t Jet_eta[maxNJets];
    Float_t Jet_pt[maxNJets];
    Float_t Jet_phi[maxNJets];
    Float_t Jet_mass[maxNJets];

    /**
     * @brief Selects the previous branches, setting their
     * status to one.
     */
    chain->SetBranchStatus("nJet", 1);
    chain->SetBranchStatus("MET_pt", 1);
    chain->SetBranchStatus("MET_phi", 1);
    chain->SetBranchStatus("MET_covXX", 1);
    chain->SetBranchStatus("MET_covXY", 1);
    chain->SetBranchStatus("MET_covYY", 1);
    chain->SetBranchStatus("MET_significance", 1);
    chain->SetBranchStatus("GenMET_pt", 1);

    chain->SetBranchStatus("Jet_area", 1);
    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
  
    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     **/
    chain->SetBranchAddress("nJet", &nJet);
    chain->SetBranchAddress("MET_pt", &MET_pt);
    chain->SetBranchAddress("MET_phi", &MET_phi);
    chain->SetBranchAddress("MET_covXX", &MET_covXX);
    chain->SetBranchAddress("MET_covXY", &MET_covXY);
    chain->SetBranchAddress("MET_covYY", &MET_covYY);
    chain->SetBranchAddress("MET_significance", &MET_significance);
    chain->SetBranchAddress("GenMET_pt", &GenMET_pt);

    chain->SetBranchAddress("Jet_area", &Jet_area);
    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    
    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);

    /**
     * @brief Selects only the first three jets and
     * defines them as new branches.
     */
    Float_t Jet_area_st, Jet_area_nd, Jet_area_rd;
    Float_t Jet_eta_st, Jet_eta_nd, Jet_eta_rd;
    Float_t Jet_pt_st, Jet_pt_nd, Jet_pt_rd;
    Float_t Jet_phi_st, Jet_phi_nd, Jet_phi_rd;
    Float_t Jet_mass_st, Jet_mass_nd, Jet_mass_rd;

    // First Jet
    newtree->Branch("Jet_area_st", &Jet_area_st);
    newtree->Branch("Jet_eta_st", &Jet_eta_st);
    newtree->Branch("Jet_pt_st", &Jet_pt_st);
    newtree->Branch("Jet_phi_st", &Jet_phi_st);
    newtree->Branch("Jet_mass_st", &Jet_mass_st);

    // Second Jet
    newtree->Branch("Jet_area_nd", &Jet_area_nd);
    newtree->Branch("Jet_eta_nd", &Jet_eta_nd);
    newtree->Branch("Jet_pt_nd", &Jet_pt_nd);
    newtree->Branch("Jet_phi_nd", &Jet_phi_nd);
    newtree->Branch("Jet_mass_nd", &Jet_mass_nd);

    // Third Jet
    newtree->Branch("Jet_area_rd", &Jet_area_rd);
    newtree->Branch("Jet_eta_rd", &Jet_eta_rd);
    newtree->Branch("Jet_pt_rd", &Jet_pt_rd);
    newtree->Branch("Jet_phi_rd", &Jet_phi_rd);
    newtree->Branch("Jet_mass_rd", &Jet_mass_rd);

    Float_t max_nJet;

    Long64_t n_events = chain->GetEntries();

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);

        // Max number of Jets
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }

        // First Jet
        Jet_area_st = (nJet > 0) ? Jet_area[0] : 0.0f;
        Jet_eta_st = (nJet > 0) ? Jet_eta[0] : 0.0f;
        Jet_pt_st = (nJet > 0) ? Jet_pt[0] : 0.0f;
        Jet_phi_st = (nJet > 0) ? Jet_phi[0] : 0.0f;
        Jet_mass_st = (nJet > 0) ? Jet_mass[0] : 0.0f;
        
        // Second Jet
        Jet_area_nd = (nJet > 1) ? Jet_area[1] : 0.0f;
        Jet_eta_nd = (nJet > 1) ? Jet_eta[1] : 0.0f;
        Jet_pt_nd = (nJet > 1) ? Jet_pt[1] : 0.0f;
        Jet_phi_nd = (nJet > 1) ? Jet_phi[1] : 0.0f;
        Jet_mass_nd = (nJet > 1) ? Jet_mass[1] : 0.0f;
        
        // Third Jet
        Jet_area_rd = (nJet > 2) ? Jet_area[2] : 0.0f;
        Jet_eta_rd = (nJet > 2) ? Jet_eta[2] : 0.0f;
        Jet_pt_rd = (nJet > 2) ? Jet_pt[2] : 0.0f;
        Jet_phi_rd = (nJet > 2) ? Jet_phi[2] : 0.0f;
        Jet_mass_rd = (nJet > 2) ? Jet_mass[2] : 0.0f;

        // Fill Tree with new entries
        newtree->Fill();
    }

    std::cout << "Max number of Jets is: " << max_nJet;

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/skimmed_ZZTo2L2Nu_0.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}