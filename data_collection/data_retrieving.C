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
    TFile *file = TFile::Open("../OriginalDatasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");
    //TFile *file = TFile::Open("../skimmed_datasets/skimmed_ZZZ.root");
    //TFile *file = TFile::Open("../skimmed_datasets/specific_skimmed_ZZZ.root");
    
    //TFile *file = TFile::Open("../datasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");
    //TFile *file = TFile::Open("../skimmed_datasets/skimmed_HToAATo2Mu2B.root");
    //TFile *file = TFile::Open("../skimmed_datasets/specific_skimmed_HToAATo2Mu2B_try.root");
    
    //TFile *file = TFile::Open("../datasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");
    //TFile *file = TFile::Open("../skimmed_datasets/skimmed_ZZTo2L2Nu.root");
    //TFile *file = TFile::Open("../skimmed_datasets/specific_skimmed_ZZTo2L2Nu.root");

    //TFile *file = TFile::Open("../SkimmedDatasets/skimmed_ZZTo2L2Nu.root");

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
    TTree *event = (TTree*)file->Get("Events");

    if (!event) {
        std::cerr << "Error loading TTree 'Events'" << std::endl;
        file->Close();
        exit(-1);
    }

    event->Show(13);



    /**
     * @brief Calculates the number of events.
     */
    TTree *tree = (TTree*)file->Get("Events");
    Long64_t n_events = tree->GetEntries();

    std::cout << "nEvents:" << n_events << std::endl;



    /**
     * @brief Gets number of branches inside the 'Events' TTree.
     */
    int nBranches = event->GetListOfBranches()->GetEntries();
    std::cout << "Number of branches in Events: " << nBranches << std::endl;



    /**
     * @brief Closes file.
     */
    file->Close();
}

