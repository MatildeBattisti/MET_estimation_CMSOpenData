#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>
#include <vector>
#include <cmath>



/**
 * @brief Defines single value input global variables:
 * - the entries have name and type of the corresponding Branches;
 * - the entries Branch status is set to one;
 * - gets the address of the Branches in order to copy their values.
 */
struct Branches {
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

        chain->SetBranchAddress("MET_pt",          &MET_pt);
        chain->SetBranchAddress("MET_phi",         &MET_phi);
        chain->SetBranchAddress("MET_covXX",       &MET_covXX);
        chain->SetBranchAddress("MET_covXY",       &MET_covXY);
        chain->SetBranchAddress("MET_covYY",       &MET_covYY);
        chain->SetBranchAddress("MET_significance",&MET_significance);
        chain->SetBranchAddress("GenMET_pt",       &GenMET_pt);
        chain->SetBranchAddress("PV_chi2",         &PV_chi2);
        chain->SetBranchAddress("PV_score",        &PV_score);
        chain->SetBranchAddress("PV_x",            &PV_x);
        chain->SetBranchAddress("PV_y",            &PV_y);
        chain->SetBranchAddress("PV_z",            &PV_z);
        chain->SetBranchAddress("nSV",             &nSV);
        chain->SetBranchAddress("nElectron",       &nElectron);
        chain->SetBranchAddress("nMuon",           &nMuon);
    }
};



/**
 * @brief Gets the maximum number of leptons across all events.
 */
struct MaxLeptons {
    UInt_t nElectron;
    UInt_t nMuon;
};

MaxLeptons getMaxLeptons(TChain* chain, Long64_t n_events,
                         UInt_t& nElectron, UInt_t& nMuon) {
    MaxLeptons max = {0, 0};
    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (nElectron > max.nElectron) max.nElectron = nElectron;
        if (nMuon     > max.nMuon)     max.nMuon     = nMuon;
    }
    return max;
}



/**
 * @brief Calculates mean and standard deviation of GenMET_pt.
 */
std::pair<Float_t, Float_t> computeMeanAndStdGenMET(TChain* chain, Long64_t n_events, Float_t& GenMET_pt) {
    Float_t sum = 0.0;
    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        sum += GenMET_pt;
    }
    Float_t mean = sum / n_events;

    Float_t var_num = 0.0;
    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        var_num += (GenMET_pt - mean) * (GenMET_pt - mean);
    }
    return {mean, std::sqrt(var_num / n_events)};
}



/**
 * @brief Computes the q-th quantile of a sorted vector using linear interpolation.
 */
Float_t computeQuantile(const std::vector<Float_t>& sorted_values, Float_t q) {
    size_t n = sorted_values.size();
    Float_t pos = q * (n - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = lo + 1;
    Float_t frac = pos - lo;

    if (hi >= n) return sorted_values[n - 1];
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]);
}



/**
 * @brief Computes the upper and lower boundaries for outlier removal
 * using the IQR method.
 */
std::pair<Float_t, Float_t> computeIQRBoundaries(TChain* chain, Long64_t n_events, Float_t& branch) {
    std::vector<Float_t> values;
    values.reserve(n_events);

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);
        if (std::isfinite(branch)) values.push_back(branch);
    }

    std::sort(values.begin(), values.end());

    Float_t q1  = computeQuantile(values, 0.25f);
    Float_t q3  = computeQuantile(values, 0.75f);
    Float_t iqr = q3 - q1;

    return {q1 - 1.5f * iqr, q3 + 1.5f * iqr};
}



/**
 * @brief Output variables – up to 6 slots per flavour.
 *
 * Naming convention: _st/_nd/_rd/_4th/_5th/_6th
 *
 * For channels with fewer than 6 charged leptons the missing slots are
 * zero-padded; the neutrino multiplicity can be inferred as:
 *   n_nu = 6 - nElectron - nMuon
 *
 * Valid ZZZ inclusive channels (nE + nMu even, 0 <= nE+nMu <= 6):
 *
 *   nE+nMu = 0  ->  0e + 0mu + 6nu   (fully invisible)
 *   nE+nMu = 2  ->  2e+0mu+4nu  |  0e+2mu+4nu   (one Z visible)
 *   nE+nMu = 4  ->  4e+0mu+2nu  |  2e+2mu+2nu  |  0e+4mu+2nu   (two Z visible)
 *   nE+nMu = 6  ->  6e  |  4e+2mu  |  2e+4mu  |  6mu   (fully leptonic)
 */

// --- Electrons (slots 0-5) ---
Int_t   Electron_charge_st,  Electron_charge_nd,  Electron_charge_rd,
        Electron_charge_4th, Electron_charge_5th, Electron_charge_6th;
Float_t Electron_dxy_st,     Electron_dxy_nd,     Electron_dxy_rd,
        Electron_dxy_4th,    Electron_dxy_5th,    Electron_dxy_6th;
Float_t Electron_dz_st,      Electron_dz_nd,      Electron_dz_rd,
        Electron_dz_4th,     Electron_dz_5th,     Electron_dz_6th;
Float_t Electron_eta_st,     Electron_eta_nd,     Electron_eta_rd,
        Electron_eta_4th,    Electron_eta_5th,    Electron_eta_6th;
Float_t Electron_mass_st,    Electron_mass_nd,    Electron_mass_rd,
        Electron_mass_4th,   Electron_mass_5th,   Electron_mass_6th;
Float_t Electron_phi_st,     Electron_phi_nd,     Electron_phi_rd,
        Electron_phi_4th,    Electron_phi_5th,    Electron_phi_6th;
Float_t Electron_pt_st,      Electron_pt_nd,      Electron_pt_rd,
        Electron_pt_4th,     Electron_pt_5th,     Electron_pt_6th;

// --- Muons (slots 0-5) ---
Int_t   Muon_charge_st,  Muon_charge_nd,  Muon_charge_rd,
        Muon_charge_4th, Muon_charge_5th, Muon_charge_6th;
Float_t Muon_dxy_st,     Muon_dxy_nd,     Muon_dxy_rd,
        Muon_dxy_4th,    Muon_dxy_5th,    Muon_dxy_6th;
Float_t Muon_dz_st,      Muon_dz_nd,      Muon_dz_rd,
        Muon_dz_4th,     Muon_dz_5th,     Muon_dz_6th;
Float_t Muon_eta_st,     Muon_eta_nd,     Muon_eta_rd,
        Muon_eta_4th,    Muon_eta_5th,    Muon_eta_6th;
Float_t Muon_mass_st,    Muon_mass_nd,    Muon_mass_rd,
        Muon_mass_4th,   Muon_mass_5th,   Muon_mass_6th;
Float_t Muon_phi_st,     Muon_phi_nd,     Muon_phi_rd,
        Muon_phi_4th,    Muon_phi_5th,    Muon_phi_6th;
Float_t Muon_pt_st,      Muon_pt_nd,      Muon_pt_rd,
        Muon_pt_4th,     Muon_pt_5th,     Muon_pt_6th;

/**
 * @brief Inferred neutrino multiplicity stored as a branch for downstream use.
 * n_nu = 6 - nElectron - nMuon
 */
Int_t n_nu;



/**
 * @brief Fills one electron slot (0-5) from flat arrays.
 */
void set_electron(int slot,
                  const Int_t*   charge, const Float_t* dxy,  const Float_t* dz,
                  const Float_t* eta,   const Float_t* mass, const Float_t* phi,
                  const Float_t* pt) {
    switch (slot) {
        case 0: Electron_charge_st  = charge[0]; Electron_dxy_st  = dxy[0]; Electron_dz_st  = dz[0]; Electron_eta_st  = eta[0]; Electron_mass_st  = mass[0]; Electron_phi_st  = phi[0]; Electron_pt_st  = pt[0]; break;
        case 1: Electron_charge_nd  = charge[1]; Electron_dxy_nd  = dxy[1]; Electron_dz_nd  = dz[1]; Electron_eta_nd  = eta[1]; Electron_mass_nd  = mass[1]; Electron_phi_nd  = phi[1]; Electron_pt_nd  = pt[1]; break;
        case 2: Electron_charge_rd  = charge[2]; Electron_dxy_rd  = dxy[2]; Electron_dz_rd  = dz[2]; Electron_eta_rd  = eta[2]; Electron_mass_rd  = mass[2]; Electron_phi_rd  = phi[2]; Electron_pt_rd  = pt[2]; break;
        case 3: Electron_charge_4th = charge[3]; Electron_dxy_4th = dxy[3]; Electron_dz_4th = dz[3]; Electron_eta_4th = eta[3]; Electron_mass_4th = mass[3]; Electron_phi_4th = phi[3]; Electron_pt_4th = pt[3]; break;
        case 4: Electron_charge_5th = charge[4]; Electron_dxy_5th = dxy[4]; Electron_dz_5th = dz[4]; Electron_eta_5th = eta[4]; Electron_mass_5th = mass[4]; Electron_phi_5th = phi[4]; Electron_pt_5th = pt[4]; break;
        case 5: Electron_charge_6th = charge[5]; Electron_dxy_6th = dxy[5]; Electron_dz_6th = dz[5]; Electron_eta_6th = eta[5]; Electron_mass_6th = mass[5]; Electron_phi_6th = phi[5]; Electron_pt_6th = pt[5]; break;
    }
}

/**
 * @brief Fills one muon slot (0-5) from flat arrays.
 */
void set_muon(int slot,
              const Int_t*   charge, const Float_t* dxy,  const Float_t* dz,
              const Float_t* eta,   const Float_t* mass, const Float_t* phi,
              const Float_t* pt) {
    switch (slot) {
        case 0: Muon_charge_st  = charge[0]; Muon_dxy_st  = dxy[0]; Muon_dz_st  = dz[0]; Muon_eta_st  = eta[0]; Muon_mass_st  = mass[0]; Muon_phi_st  = phi[0]; Muon_pt_st  = pt[0]; break;
        case 1: Muon_charge_nd  = charge[1]; Muon_dxy_nd  = dxy[1]; Muon_dz_nd  = dz[1]; Muon_eta_nd  = eta[1]; Muon_mass_nd  = mass[1]; Muon_phi_nd  = phi[1]; Muon_pt_nd  = pt[1]; break;
        case 2: Muon_charge_rd  = charge[2]; Muon_dxy_rd  = dxy[2]; Muon_dz_rd  = dz[2]; Muon_eta_rd  = eta[2]; Muon_mass_rd  = mass[2]; Muon_phi_rd  = phi[2]; Muon_pt_rd  = pt[2]; break;
        case 3: Muon_charge_4th = charge[3]; Muon_dxy_4th = dxy[3]; Muon_dz_4th = dz[3]; Muon_eta_4th = eta[3]; Muon_mass_4th = mass[3]; Muon_phi_4th = phi[3]; Muon_pt_4th = pt[3]; break;
        case 4: Muon_charge_5th = charge[4]; Muon_dxy_5th = dxy[4]; Muon_dz_5th = dz[4]; Muon_eta_5th = eta[4]; Muon_mass_5th = mass[4]; Muon_phi_5th = phi[4]; Muon_pt_5th = pt[4]; break;
        case 5: Muon_charge_6th = charge[5]; Muon_dxy_6th = dxy[5]; Muon_dz_6th = dz[5]; Muon_eta_6th = eta[5]; Muon_mass_6th = mass[5]; Muon_phi_6th = phi[5]; Muon_pt_6th = pt[5]; break;
    }
}

void empty_electron_variables() {
    Electron_charge_st = 0; Electron_charge_nd = 0; Electron_charge_rd = 0;
    Electron_charge_4th = 0; Electron_charge_5th = 0; Electron_charge_6th = 0;
    Electron_dxy_st = 0; Electron_dxy_nd = 0; Electron_dxy_rd = 0;
    Electron_dxy_4th = 0; Electron_dxy_5th = 0; Electron_dxy_6th = 0;
    Electron_dz_st = 0; Electron_dz_nd = 0; Electron_dz_rd = 0;
    Electron_dz_4th = 0; Electron_dz_5th = 0; Electron_dz_6th = 0;
    Electron_eta_st = 0; Electron_eta_nd = 0; Electron_eta_rd = 0;
    Electron_eta_4th = 0; Electron_eta_5th = 0; Electron_eta_6th = 0;
    Electron_mass_st = 0; Electron_mass_nd = 0; Electron_mass_rd = 0;
    Electron_mass_4th = 0; Electron_mass_5th = 0; Electron_mass_6th = 0;
    Electron_phi_st = 0; Electron_phi_nd = 0; Electron_phi_rd = 0;
    Electron_phi_4th = 0; Electron_phi_5th = 0; Electron_phi_6th = 0;
    Electron_pt_st = 0; Electron_pt_nd = 0; Electron_pt_rd = 0;
    Electron_pt_4th = 0; Electron_pt_5th = 0; Electron_pt_6th = 0;
}

void empty_muon_variables() {
    Muon_charge_st = 0; Muon_charge_nd = 0; Muon_charge_rd = 0;
    Muon_charge_4th = 0; Muon_charge_5th = 0; Muon_charge_6th = 0;
    Muon_dxy_st = 0; Muon_dxy_nd = 0; Muon_dxy_rd = 0;
    Muon_dxy_4th = 0; Muon_dxy_5th = 0; Muon_dxy_6th = 0;
    Muon_dz_st = 0; Muon_dz_nd = 0; Muon_dz_rd = 0;
    Muon_dz_4th = 0; Muon_dz_5th = 0; Muon_dz_6th = 0;
    Muon_eta_st = 0; Muon_eta_nd = 0; Muon_eta_rd = 0;
    Muon_eta_4th = 0; Muon_eta_5th = 0; Muon_eta_6th = 0;
    Muon_mass_st = 0; Muon_mass_nd = 0; Muon_mass_rd = 0;
    Muon_mass_4th = 0; Muon_mass_5th = 0; Muon_mass_6th = 0;
    Muon_phi_st = 0; Muon_phi_nd = 0; Muon_phi_rd = 0;
    Muon_phi_4th = 0; Muon_phi_5th = 0; Muon_phi_6th = 0;
    Muon_pt_st = 0; Muon_pt_nd = 0; Muon_pt_rd = 0;
    Muon_pt_4th = 0; Muon_pt_5th = 0; Muon_pt_6th = 0;
}



/**
 * @brief Main.
 */
void preparation_ZZZ() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("../OriginalDatasets/ZZZ/47348ED1-E550-CF48-9E94-BED2742AB141.root");



    /**
     * @param n_events Number of events in the file.
     */
    Long64_t n_events = chain->GetEntries();
    std::cout << "nEvents before skimming: " << n_events << std::endl;



    /**
     * @brief Sets up the single value input branches.
     */
    Branches branches;
    branches.setup_branches(chain.get());



    /**
     * @brief Gets the maximum number of leptons across all events.
     */
    auto max = getMaxLeptons(chain.get(), n_events,
                             branches.nElectron, branches.nMuon);
    std::cout << "Max nElectron: " << max.nElectron << std::endl;
    std::cout << "Max nMuon:     " << max.nMuon     << std::endl;



    /**
     * @brief Calculates GenMET_pt mean and standard deviation.
     */
    auto [GenMET_mean, GenMET_std] = computeMeanAndStdGenMET(chain.get(), n_events, branches.GenMET_pt);
    std::cout << "Mean GenMET_pt:               " << GenMET_mean << std::endl;
    std::cout << "Standard deviation GenMET_pt: " << GenMET_std  << std::endl;



    /**
     * @brief Defines variables for Branches with variable-length arrays.
     *
     * Vectors are sized to the global maximum so ROOT always writes into
     * a valid memory region; only the first nElectron/nMuon elements are
     * meaningful for a given event.
     */
    std::vector<Float_t> Electron_dxy(max.nElectron);
    std::vector<Float_t> Electron_dz(max.nElectron);
    std::vector<Float_t> Electron_eta(max.nElectron);
    std::vector<Float_t> Electron_mass(max.nElectron);
    std::vector<Float_t> Electron_phi(max.nElectron);
    std::vector<Float_t> Electron_pt(max.nElectron);
    std::vector<Int_t>   Electron_charge(max.nElectron);

    std::vector<Float_t> Muon_dxy(max.nMuon);
    std::vector<Float_t> Muon_dz(max.nMuon);
    std::vector<Float_t> Muon_eta(max.nMuon);
    std::vector<Float_t> Muon_mass(max.nMuon);
    std::vector<Float_t> Muon_phi(max.nMuon);
    std::vector<Float_t> Muon_pt(max.nMuon);
    std::vector<Int_t>   Muon_charge(max.nMuon);

    chain->SetBranchStatus("Electron_dxy",    1);
    chain->SetBranchStatus("Electron_dz",     1);
    chain->SetBranchStatus("Electron_eta",    1);
    chain->SetBranchStatus("Electron_mass",   1);
    chain->SetBranchStatus("Electron_phi",    1);
    chain->SetBranchStatus("Electron_pt",     1);
    chain->SetBranchStatus("Electron_charge", 1);

    chain->SetBranchStatus("Muon_dxy",    1);
    chain->SetBranchStatus("Muon_dz",     1);
    chain->SetBranchStatus("Muon_eta",    1);
    chain->SetBranchStatus("Muon_mass",   1);
    chain->SetBranchStatus("Muon_phi",    1);
    chain->SetBranchStatus("Muon_pt",     1);
    chain->SetBranchStatus("Muon_charge", 1);

    chain->SetBranchAddress("Electron_dxy",    Electron_dxy.data());
    chain->SetBranchAddress("Electron_dz",     Electron_dz.data());
    chain->SetBranchAddress("Electron_eta",    Electron_eta.data());
    chain->SetBranchAddress("Electron_mass",   Electron_mass.data());
    chain->SetBranchAddress("Electron_phi",    Electron_phi.data());
    chain->SetBranchAddress("Electron_pt",     Electron_pt.data());
    chain->SetBranchAddress("Electron_charge", Electron_charge.data());

    chain->SetBranchAddress("Muon_dxy",    Muon_dxy.data());
    chain->SetBranchAddress("Muon_dz",     Muon_dz.data());
    chain->SetBranchAddress("Muon_eta",    Muon_eta.data());
    chain->SetBranchAddress("Muon_mass",   Muon_mass.data());
    chain->SetBranchAddress("Muon_phi",    Muon_phi.data());
    chain->SetBranchAddress("Muon_pt",     Muon_pt.data());
    chain->SetBranchAddress("Muon_charge", Muon_charge.data());



    /**
     * @brief Clone full TTree structure (without entries).
     */
    TTree* newtree = chain->CloneTree(0);



    /**
     * @brief Add output branches – six lepton slots per flavour plus n_nu.
     *
     * n_nu = 6 - nElectron - nMuon is saved explicitly so downstream
     * analyses can select on neutrino multiplicity without recomputing it,
     * and apply MET-based requirements accordingly.
     */
    newtree->Branch("n_nu", &n_nu);

    // Electrons
    newtree->Branch("Electron_charge_st",  &Electron_charge_st);
    newtree->Branch("Electron_dxy_st",     &Electron_dxy_st);
    newtree->Branch("Electron_dz_st",      &Electron_dz_st);
    newtree->Branch("Electron_eta_st",     &Electron_eta_st);
    newtree->Branch("Electron_mass_st",    &Electron_mass_st);
    newtree->Branch("Electron_phi_st",     &Electron_phi_st);
    newtree->Branch("Electron_pt_st",      &Electron_pt_st);

    newtree->Branch("Electron_charge_nd",  &Electron_charge_nd);
    newtree->Branch("Electron_dxy_nd",     &Electron_dxy_nd);
    newtree->Branch("Electron_dz_nd",      &Electron_dz_nd);
    newtree->Branch("Electron_eta_nd",     &Electron_eta_nd);
    newtree->Branch("Electron_mass_nd",    &Electron_mass_nd);
    newtree->Branch("Electron_phi_nd",     &Electron_phi_nd);
    newtree->Branch("Electron_pt_nd",      &Electron_pt_nd);

    newtree->Branch("Electron_charge_rd",  &Electron_charge_rd);
    newtree->Branch("Electron_dxy_rd",     &Electron_dxy_rd);
    newtree->Branch("Electron_dz_rd",      &Electron_dz_rd);
    newtree->Branch("Electron_eta_rd",     &Electron_eta_rd);
    newtree->Branch("Electron_mass_rd",    &Electron_mass_rd);
    newtree->Branch("Electron_phi_rd",     &Electron_phi_rd);
    newtree->Branch("Electron_pt_rd",      &Electron_pt_rd);

    newtree->Branch("Electron_charge_4th", &Electron_charge_4th);
    newtree->Branch("Electron_dxy_4th",    &Electron_dxy_4th);
    newtree->Branch("Electron_dz_4th",     &Electron_dz_4th);
    newtree->Branch("Electron_eta_4th",    &Electron_eta_4th);
    newtree->Branch("Electron_mass_4th",   &Electron_mass_4th);
    newtree->Branch("Electron_phi_4th",    &Electron_phi_4th);
    newtree->Branch("Electron_pt_4th",     &Electron_pt_4th);

    newtree->Branch("Electron_charge_5th", &Electron_charge_5th);
    newtree->Branch("Electron_dxy_5th",    &Electron_dxy_5th);
    newtree->Branch("Electron_dz_5th",     &Electron_dz_5th);
    newtree->Branch("Electron_eta_5th",    &Electron_eta_5th);
    newtree->Branch("Electron_mass_5th",   &Electron_mass_5th);
    newtree->Branch("Electron_phi_5th",    &Electron_phi_5th);
    newtree->Branch("Electron_pt_5th",     &Electron_pt_5th);

    newtree->Branch("Electron_charge_6th", &Electron_charge_6th);
    newtree->Branch("Electron_dxy_6th",    &Electron_dxy_6th);
    newtree->Branch("Electron_dz_6th",     &Electron_dz_6th);
    newtree->Branch("Electron_eta_6th",    &Electron_eta_6th);
    newtree->Branch("Electron_mass_6th",   &Electron_mass_6th);
    newtree->Branch("Electron_phi_6th",    &Electron_phi_6th);
    newtree->Branch("Electron_pt_6th",     &Electron_pt_6th);

    // Muons
    newtree->Branch("Muon_charge_st",  &Muon_charge_st);
    newtree->Branch("Muon_dxy_st",     &Muon_dxy_st);
    newtree->Branch("Muon_dz_st",      &Muon_dz_st);
    newtree->Branch("Muon_eta_st",     &Muon_eta_st);
    newtree->Branch("Muon_mass_st",    &Muon_mass_st);
    newtree->Branch("Muon_phi_st",     &Muon_phi_st);
    newtree->Branch("Muon_pt_st",      &Muon_pt_st);

    newtree->Branch("Muon_charge_nd",  &Muon_charge_nd);
    newtree->Branch("Muon_dxy_nd",     &Muon_dxy_nd);
    newtree->Branch("Muon_dz_nd",      &Muon_dz_nd);
    newtree->Branch("Muon_eta_nd",     &Muon_eta_nd);
    newtree->Branch("Muon_mass_nd",    &Muon_mass_nd);
    newtree->Branch("Muon_phi_nd",     &Muon_phi_nd);
    newtree->Branch("Muon_pt_nd",      &Muon_pt_nd);

    newtree->Branch("Muon_charge_rd",  &Muon_charge_rd);
    newtree->Branch("Muon_dxy_rd",     &Muon_dxy_rd);
    newtree->Branch("Muon_dz_rd",      &Muon_dz_rd);
    newtree->Branch("Muon_eta_rd",     &Muon_eta_rd);
    newtree->Branch("Muon_mass_rd",    &Muon_mass_rd);
    newtree->Branch("Muon_phi_rd",     &Muon_phi_rd);
    newtree->Branch("Muon_pt_rd",      &Muon_pt_rd);

    newtree->Branch("Muon_charge_4th", &Muon_charge_4th);
    newtree->Branch("Muon_dxy_4th",    &Muon_dxy_4th);
    newtree->Branch("Muon_dz_4th",     &Muon_dz_4th);
    newtree->Branch("Muon_eta_4th",    &Muon_eta_4th);
    newtree->Branch("Muon_mass_4th",   &Muon_mass_4th);
    newtree->Branch("Muon_phi_4th",    &Muon_phi_4th);
    newtree->Branch("Muon_pt_4th",     &Muon_pt_4th);

    newtree->Branch("Muon_charge_5th", &Muon_charge_5th);
    newtree->Branch("Muon_dxy_5th",    &Muon_dxy_5th);
    newtree->Branch("Muon_dz_5th",     &Muon_dz_5th);
    newtree->Branch("Muon_eta_5th",    &Muon_eta_5th);
    newtree->Branch("Muon_mass_5th",   &Muon_mass_5th);
    newtree->Branch("Muon_phi_5th",    &Muon_phi_5th);
    newtree->Branch("Muon_pt_5th",     &Muon_pt_5th);

    newtree->Branch("Muon_charge_6th", &Muon_charge_6th);
    newtree->Branch("Muon_dxy_6th",    &Muon_dxy_6th);
    newtree->Branch("Muon_dz_6th",     &Muon_dz_6th);
    newtree->Branch("Muon_eta_6th",    &Muon_eta_6th);
    newtree->Branch("Muon_mass_6th",   &Muon_mass_6th);
    newtree->Branch("Muon_phi_6th",    &Muon_phi_6th);
    newtree->Branch("Muon_pt_6th",     &Muon_pt_6th);



    /**
     * @brief Event loop – inclusive ZZZ selection.
     *
     * Selection criteria:
     *   1. nElectron + nMuon <= 6   (cannot exceed total Z decay products)
     *   2. nElectron + nMuon is even (charge conservation across all three Z)
     *
     * The neutrino multiplicity n_nu = 6 - nE - nMu is saved as a branch.
     * Channels with n_nu > 0 carry significant MET (already in the cloned
     * MET_pt branch) which can be used for downstream signal/background
     * discrimination.
     */
    Long64_t n_events_skimmed = 0;

    // Channel counters indexed by [nElectron][nMuon]
    Long64_t count[7][7] = {};

    for (Long64_t i = 0; i < n_events; i++) {
        chain->GetEntry(i);

        UInt_t nE = branches.nElectron;
        UInt_t nM = branches.nMuon;

        if (nE + nM > 6)        continue;   // too many charged leptons
        if ((nE + nM) % 2 != 0) continue;   // odd total impossible from Z pairs

        n_nu = static_cast<Int_t>(6 - nE - nM);

        empty_electron_variables();
        for (UInt_t k = 0; k < nE; k++) {
            set_electron(k,
                Electron_charge.data(), Electron_dxy.data(), Electron_dz.data(),
                Electron_eta.data(),   Electron_mass.data(), Electron_phi.data(),
                Electron_pt.data());
        }

        empty_muon_variables();
        for (UInt_t k = 0; k < nM; k++) {
            set_muon(k,
                Muon_charge.data(), Muon_dxy.data(), Muon_dz.data(),
                Muon_eta.data(),   Muon_mass.data(), Muon_phi.data(),
                Muon_pt.data());
        }

        count[nE][nM]++;
        newtree->Fill();
        n_events_skimmed++;
    }



    /**
     * @brief Summary printout – total and per sub-channel.
     */
    std::cout << "\n--- Skimming results (inclusive ZZZ) ---" << std::endl;
    std::cout << "Remaining events after skimming: " << n_events_skimmed << std::endl;
    std::cout << "\nChannel breakdown (nE + nMu + nNu = 6):" << std::endl;
    std::cout << "  nE  nMu  nNu    events" << std::endl;
    for (int e = 0; e <= 6; e++) {
        for (int m = 0; m <= 6 - e; m++) {
            if ((e + m) % 2 != 0) continue;
            if (count[e][m] == 0) continue;
            std::cout << "   " << e << "    " << m << "    " << (6 - e - m)
                      << "   " << count[e][m] << std::endl;
        }
    }



    /**
     * @brief Creates output file and writes the skimmed tree.
     */
    auto skimfile = std::make_unique<TFile>("../CleanedDatasets/cleaned_ZZZInclusive.root", "RECREATE");
    newtree->Write();
    skimfile->Close();
}