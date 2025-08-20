#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void skimming_flags() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("../datasets/ZZZ/2E96A5E9-C938-A149-BBBF-8FD81A9E5AD6.root");  // dataset 0

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