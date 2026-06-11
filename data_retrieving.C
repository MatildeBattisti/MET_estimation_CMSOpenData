/**
 * @file data_retrieving.C
 * @brief Opens .root files and reads the branches inside the
 * "Events" tree.
 * Also shows the number of events and branches for each event.
 */
#include <iostream>
#include <TFile.h>
#include <TTree.h>

void data_retrieving() {
    /**
     * @brief Open file from local path.
     * Returns an error if file is not opened.
     */
    //TFile *file = TFile::Open("OriginalDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");
    TFile *file = TFile::Open("OriginalDatasets/HToAATo2Mu2B/merged_HToAATo2Mu2B.root");
    //TFile *file = TFile::Open("/OriginalDdatasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");

    //TFile *file = TFile::Open("SkimmedDatasets/skimmed_HToAATo2Mu2B.root");
    //TFile *file = TFile::Open("SkimmedDatasets/skimmed_ZZTo2L2Nu.root");

    //TFile *file = TFile::Open("CleanedDatasets/cleaned_HToAATo2Mu2B.root");
    //TFile *file = TFile::Open("CleanedDatasets/cleaned_ZZTo2L2Nu.root");

    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file." << std::endl;
        exit(-1);
     }
    file->ls();
    


    /**
     * @brief Get 'Events' tree entries.
     * If Show() is left empty, shows EVENT:-1. We only use it to see the entries name.
     * Returns error if the tree isn't loaded.
     */
    TTree *tree = (TTree*)file->Get("Events");

    if (!tree) {
        std::cerr << "Error loading TTree 'Events'" << std::endl;
        file->Close();
        exit(-1);
    }

    tree->Show(1);



    /**
     * @brief Prints number of events and branches.
     */
    Long64_t n_events  = tree->GetEntries();
    int nBranches = tree->GetListOfBranches()->GetEntries();

    std::cout << "nEvents:" << n_events << std::endl;
    std::cout << "Number of branches in Events: " << nBranches << std::endl;



    /**
     * @brief Closes file.
     */
    file->Close();
}
