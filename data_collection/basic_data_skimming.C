/**
 * @file basic_data_skimming.C
 * @brief Skims the original datasets by only using the branhes that are:
 * - relevant to the Ml model for the MET resolution;
 * - significant for all the three different datasets.
 */
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void basic_data_skimming() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    //chain->Add("../datasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");  // ZZZ

    chain->Add("../datasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");  // HToAATo2Mu2B
   
    //chain->Add("../datasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");  // ZZTo2L2Nu



    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);



    /**
     * @brief Only selects entries that are relevant for all
     * three types of dataset for the ML model.
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

    Float_t PV_chi2;
    Float_t PV_score;
    Float_t PV_x;
    Float_t PV_y;
    Float_t PV_z;

    UInt_t nSV;



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
    chain->SetBranchStatus("PV_chi2", 1);
    chain->SetBranchStatus("PV_score", 1);
    chain->SetBranchStatus("PV_x", 1);
    chain->SetBranchStatus("PV_y", 1);
    chain->SetBranchStatus("PV_z", 1);
    chain->SetBranchStatus("nSV", 1);
  


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
    chain->SetBranchAddress("PV_chi2", &PV_chi2);
    chain->SetBranchAddress("PV_score", &PV_score);
    chain->SetBranchAddress("PV_x", &PV_x);
    chain->SetBranchAddress("PV_y", &PV_y);
    chain->SetBranchAddress("PV_z", &PV_z);
    chain->SetBranchAddress("nSV", &nSV);
    


    /**
     * @param n_events Number of events in each file
     */
    Long64_t n_events = chain->GetEntries();

    std::cout << "Number of events:" << n_events << std::endl; 



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

    std::cout << "Max number of Jets is: " << max_nJet << std::endl;



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
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/skimmed_HToAATo2Mu2B.root", "RECREATE");



    /**
     * @brief Clone full TTree structure and content,
     * considering that we set some branches status to zero.
     */
    TTree *newtree = chain->CloneTree();



    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}