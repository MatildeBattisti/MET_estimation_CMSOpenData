/**
 * @file merge_datasets.C
 * @brief Merges .root files corresponding to the same event.
 */
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TClass.h>
#include <TFileMerger.h>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Returns the name of all TTree in the file
 */
std::vector<std::string> GetTreeNames(const std::string& filename)
{
    std::vector<std::string> names;
    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) return names;

    TIter next(f->GetListOfKeys());
    TKey* key;
    while ((key = (TKey*)next())) {
        TClass* cl = TClass::GetClass(key->GetClassName());
        if (cl && cl->InheritsFrom("TTree"))
            names.push_back(key->GetName());
    }
    f->Close();
    return names;
}



/**
 * @brief Main.
 */
void merge_datasets(){
    const std::string file1  = "OriginalDatasets/HToAATo2Mu2B/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root";
    const std::string file2  = "OriginalDatasets/HToAATo2Mu2B/6357E7BC-502C-2E45-A649-73A57B651715.root";
    const std::string output = "OriginalDatasets/HToAATo2Mu2B/augmented_HToAATo2Mu2B.root";



    /**
     * @brief Gets the names of the TTrees in the first file.
     */
    auto treeNames = GetTreeNames(file1);
    if (treeNames.empty()) {
        std::cerr << "[ERRORE] Nessun TTree trovato in " << file1 << std::endl;
        return;
    }



    /**
     * @brief For each TTree concatenates the two files.
     * The first Snapshot generates the file, the following two append.
     */
    bool firstTree = true;
    for (const auto& treeName : treeNames) {
        std::cout << "[INFO] Processing TTree: " << treeName << std::endl;

        // RDataFrame reads from both files
        ROOT::RDataFrame rdf(treeName, {file1, file2});

        ROOT::RDF::RSnapshotOptions opts;
        opts.fMode        = firstTree ? "RECREATE" : "UPDATE";
        opts.fLazy        = false;
        opts.fCompressionLevel = 1;

        // Snapshot writes the merged TTree in the output file
        rdf.Snapshot(treeName, output, rdf.GetColumnNames(), opts);

        auto nEntries = rdf.Count().GetValue();
        std::cout << " -> " << nEntries << " written entries" << std::endl;

        firstTree = false;
    }



    /**
     * @brief Using TFileMerger to copy non-TTree objects.
     * RDataFrame has already written the TTrees, now TFileMerger adds
     * the remaining objects without overriding the TTrees.
     */
    std::cout << "[INFO] Copying non-TTree objects" << std::endl;

    TFileMerger merger(kFALSE);
    merger.OutputFile(output.c_str(), "UPDATE");
    merger.AddFile(file1.c_str());
    merger.AddFile(file2.c_str());
    merger.SetNotrees(kTRUE);
    merger.Merge();

    std::cout << "\nMerging completed. Output: " << output << std::endl;
}