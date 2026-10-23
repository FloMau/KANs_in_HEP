
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, DelphesSchema
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import sys
import glob
sys.path.append("./..")
import vector
from DelphesAnalyser import event_selection
import pandas as pd
from sklearn.model_selection import train_test_split
import os

plt.style.use([hep.style.CMS])
vector.register_awkward()


def main(path, entryStop):

    print("INFO: reading file", path)
    events = NanoEventsFactory.from_root(path, treepath="Delphes", schemaclass=DelphesSchema, entry_stop=entryStop).events()
    nEvents = len(events)
    events = event_selection(events)

    print("\tINFO: number of events before & after selection:", nEvents, len(events))

    jets = ak.zip({
        "pt": events.Jet.pt,
        "eta": events.Jet.eta,
        "phi": events.Jet.phi,
        "mass": events.Jet.mass,
        "BTag": events.Jet.BTag
    }, with_name="Momentum4D")

    phoLead = ak.zip({
        "pt": events.Photon[:, 0].pt,
        "eta": events.Photon[:, 0].eta,
        "phi": events.Photon[:, 0].phi,
        "mass": events.Photon[:, 0].mass
    }, with_name="Momentum4D")

    phoSublead = ak.zip({
        "pt": events.Photon[:, 1].pt,
        "eta": events.Photon[:, 1].eta,
        "phi": events.Photon[:, 1].phi,
        "mass": events.Photon[:, 1].mass
    }, with_name="Momentum4D")

    diphoton = phoLead + phoSublead

    leptons = ak.zip({
        "pt": events.Lepton.pt,
        "eta": events.Lepton.eta,
        "phi": events.Lepton.phi,
        "mass": events.Lepton.mass,
        # "pdgId": events.Lepton.pdgId
    }, with_name="Momentum4D")

    bJets = jets[(jets.BTag == 1) & (np.abs(jets.eta) < 2.5)]

    weights = events.Weight.Weight[:,0]

    # Initialize DataFrame
    df = pd.DataFrame()

    df["phoLeadPt"] = phoLead.pt
    df["phoLeadEta"] = phoLead.eta
    df["phoLeadPhi"] = phoLead.phi

    df["phoSubleadPt"] = phoSublead.pt
    df["phoSubleadEta"] = phoSublead.eta
    df["phoSubleadPhi"] = phoSublead.phi

    # Prepare observables
    df['ptH'] = diphoton.pt.tolist()
    df['yH'] = diphoton.rapidity.tolist()
    # df['etaH'] = diphoton.eta.tolist()
    df['phiH'] = diphoton.phi.tolist()

    df['nJets'] = ak.num(jets).tolist()
    df['nLeptons'] = ak.num(leptons).tolist()
    df['nJetsCentral'] = ak.num(jets[np.abs(jets.eta)<2.5]).tolist()
    df['nBjetsCentral'] = ak.num(bJets).tolist()
    df['nJetsForward'] = ak.num(jets[np.abs(jets.eta)>2.5]).tolist()

    df['met'] = events.MissingET.MET.tolist()
    df['metPhi'] = events.MissingET.Phi.tolist()
    df['leptonPt'] = leptons[:, 0].pt.tolist()
    df['leptonEta'] = leptons[:, 0].eta.tolist()
    df['leptonPhi'] = leptons[:, 0].phi.tolist()

    # Transverse top mass
    delta_phi = (bJets[:, 0] + leptons[:, 0]).phi - events.MissingET.Phi
    topMt = np.sqrt(2 * (bJets[:, 0] + leptons[:, 0]).pt * events.MissingET.MET * (1 - np.cos(delta_phi)))
    df['topMt'] = topMt.tolist()

    # Delta y between leading b-jet and leading jet
    bJetLead = ak.firsts(bJets)

    jetLead = ak.firsts(jets[~(jets.pt == bJetLead.pt)])

    df['bJetPt'] = bJetLead.pt.tolist()
    df['bJetEta'] = bJetLead.eta.tolist()
    df['bJetPhi'] = bJetLead.phi.tolist()

    df['jetLeadPt'] = jetLead.pt.tolist()
    df['jetLeadEta'] = jetLead.eta.tolist()
    df['jetLeadPhi'] = jetLead.phi.tolist()

    df['delta_eta_b_j'] = (bJetLead.eta- jetLead.eta).tolist() 
    df['rapiditySumJH'] = np.sqrt(jetLead.rapidity**2 + diphoton.rapidity**2).tolist()
    df['rapidityDiffJH'] = (jetLead.rapidity - diphoton.rapidity).tolist()
    df['DeltaY_Hb'] = (diphoton.rapidity - bJetLead.rapidity).tolist()
    df['DeltaY_Hl'] = (diphoton.rapidity - leptons[:, 0].rapidity).tolist()

    df["weights"] = weights

    # Display the DataFrame to verify
    print(df.head())

    return df



basepath_ttH = "/net/scratch_cms3a/mausolf/MCGenMGPyDelphes/PythiaDelphes/output/ttH/"
basepath_tHq = "/net/scratch_cms3a/mausolf/MCGenMGPyDelphes/PythiaDelphes/output/tHq/"
filenamesTtHCPEven = glob.glob(basepath_ttH + "ttH*.root")
filenamesTHqCPEven = glob.glob(basepath_tHq + "tHq*.root")

stop = None

dfTtHCPEven = pd.concat(
    [main(file, entryStop=stop) for file in filenamesTtHCPEven]
)

dfTHqCPEven = pd.concat(
    [main(file, entryStop=stop) for file in filenamesTHqCPEven]
)

# Create the directory if it doesn't exist
output_dir = "./ML_datasets"
os.makedirs(output_dir, exist_ok=True)

for df, name in zip([dfTtHCPEven, dfTHqCPEven], ["ttHCPEven", "tHqCPEven"]):

    print(name, len(df))

    # First split: train set and temp set
    train_df, temp_df = train_test_split(df, test_size=4/7, random_state=42)

    # Second split: validation set and test set
    val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42)

    # Save the DataFrames to CSV files
    train_df.to_csv(os.path.join(output_dir, f'{name}_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, f'{name}_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, f'{name}_test.csv'), index=False)

    print(f"{name} split into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
