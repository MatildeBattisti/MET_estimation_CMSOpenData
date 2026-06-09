#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <memory>



/**
 * @brief Only selects entries that are relevant for the current dataset:
 * - the entries have name and type of the dataset Branches;
 * - the entries Branch status is set to one;
 * - gets the address of the Branches in order to copy their values.
 */
struct Branches {
    // Input branches
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
    UInt_t nElectron;
    UInt_t nMuon;
    UInt_t nJet;
    UInt_t nGenJet;

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
        chain->SetBranchStatus("nElectron", 1);
        chain->SetBranchStatus("nMuon", 1);
        chain->SetBranchStatus("nJet", 1);
        chain->SetBranchStatus("nGenJet", 1);

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
        chain->SetBranchAddress("nElectron", &nElectron);
        chain->SetBranchAddress("nMuon", &nMuon);
        chain->SetBranchAddress("nJet", &nJet);
        chain->SetBranchAddress("nGenJet", &nJet);
    }
};



/**
 * @brief Main.
 */
void skimming_data_understanding() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");

    chain->Add("../OriginalDatasets/HToAATo2Mu2B/augmented_HToAATo2Mu2B.root");



    /**
     * @param n_events Number of events in each file.
     */
    Long64_t n_events = chain->GetEntries();

    std::cout << "nEvents:" << n_events << std::endl;



    /**
     * @brief Sets up the single value input branches.
     */
    Branches branches;
    branches.setup_branches(chain.get());
    


    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     */
    auto skimfile = std::make_unique<TFile>("../SkimmedDatasets/skimmed_augmented_HToAATo2Mu2B.root", "RECREATE");



    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree();
    


    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}