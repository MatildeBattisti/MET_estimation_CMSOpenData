#include <iostream>
#include <TFile.h>
#include <TTree.h>

void data_retrieving() {
    /**
     * @brief Open file from local path.
     * Returns an error if file is not opened.
     */
    //TFile *file = TFile::Open("../datasets//HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");  // dataset 0
    //TFile *file = TFile::Open("../datasets/HToAATo2Mu2B/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root");  // dataset 1
    //TFile *file = TFile::Open("../datasets/HToAATo2Mu2B/77DB0F5B-4123-4E4B-A9D0-3CEBA8575834.root");  // dataset 2
    //TFile *file = TFile::Open("../datasets/HToAATo2Mu2B/048A040C-DA63-1949-9BA7-075371EB4296.root");  // dataset 3

    //TFile *file = TFile::Open("../datasets/ZZZ/2E96A5E9-C938-A149-BBBF-8FD81A9E5AD6.root");
    TFile *file = TFile::Open("../skimmed_datasets/skimmed_ZZZ.root");

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

    //event->Print();
    event->Show(1);

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