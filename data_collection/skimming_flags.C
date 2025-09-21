/**
 * @file skimming_flags.C
 * @brief Only skims the dataset's flag entries for a better readability.
 */
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void skimming_flags() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    //chain->Add("../datasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");
    //chain->Add("../datasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root");
    //chain->Add("../datasets/ZZTo2L2Nu/0E4250DC-CAD4-FC48-85EE-90B2A761B6B0.root");

    /**
     * @brief Reads all branches names
     */
    TObjArray *branches = chain->GetListOfBranches();
    for(int i=0; i<branches->GetEntries(); i++) {
        TBranch *branch = (TBranch*)branches->At(i);

        TString branchName = branch->GetName();

        if (branchName.BeginsWith("L1") || branchName.BeginsWith("Flag") || branchName.BeginsWith("HLT")) {
            chain->SetBranchStatus(branchName, 0);
        }
        else {
            chain->SetBranchStatus(branchName, 1);
        }
    }

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("../skimmed_datasets/flag_skimmed_ZZZ0.root", "RECREATE");

    /**
     * @brief Clones full TTree structure and content,
     * considering that we set some branches status to zero.
     */
    TTree *newtree = chain->CloneTree();

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}