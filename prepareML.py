
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import sys
import glob
sys.path.append("./..")
import vector
from NanoGenAnalyser import event_selection
import pandas as pd
from sklearn.model_selection import train_test_split
import os

plt.style.use([hep.style.CMS])
vector.register_awkward()


def main(path, entryStop):

    print("INFO: reading file", path)
    events = NanoEventsFactory.from_root(path, schemaclass=NanoAODSchema, entry_stop=entryStop).events()
    nEvents = len(events)
    events = event_selection(events)


    print("\tINFO: number of events before & after selection:", nEvents, len(events))

    genJets = ak.zip({ 
        "pt": events.GenJet.pt,
        "eta": events.GenJet.eta,
        "phi": events.GenJet.phi,
        "mass": events.GenJet.mass,
        "partonFlavour": events.GenJet.partonFlavour
    }, with_name="Momentum4D")

    phoLead = ak.zip({
        "pt": events.GenIsolatedPhoton[:, 0].pt,
        "eta": events.GenIsolatedPhoton[:, 0].eta,
        "phi": events.GenIsolatedPhoton[:, 0].phi,
        "mass": events.GenIsolatedPhoton[:, 0].mass
    }, with_name="Momentum4D")

    phoSublead = ak.zip({
        "pt": events.GenIsolatedPhoton[:, 1].pt,
        "eta": events.GenIsolatedPhoton[:, 1].eta,
        "phi": events.GenIsolatedPhoton[:, 1].phi,
        "mass": events.GenIsolatedPhoton[:, 1].mass
    }, with_name="Momentum4D")

    diphoton = phoLead + phoSublead

    leptons = ak.zip({
        "pt": events.GenDressedLepton.pt,
        "eta": events.GenDressedLepton.eta,
        "phi": events.GenDressedLepton.phi,
        "mass": events.GenDressedLepton.mass,
        "pdgId": events.GenDressedLepton.pdgId
    }, with_name="Momentum4D")

    bJets = genJets[np.abs(genJets.partonFlavour) == 5]

    weights = events.genWeight


    # Assuming 'events' and all the previous variables are already defined

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

    df['nJets'] = ak.num(events.GenJet).tolist()
    df['nLeptons'] = ak.num(leptons).tolist()
    df['nJetsCentral'] = ak.num(events.GenJet[np.abs(events.GenJet.eta)<2.5]).tolist()
    df['nBjetsCentral'] = ak.num(events.GenJet[(np.abs(events.GenJet.eta)<2.5) & (np.abs(events.GenJet.partonFlavour)==5)]).tolist()
    df['nJetsForward'] = ak.num(events.GenJet[np.abs(events.GenJet.eta)>2.5]).tolist()

    df['met'] = events.MET.fiducialGenPt.tolist()
    df['metPhi'] = events.MET.fiducialGenPhi.tolist()
    df['leptonPt'] = leptons[:, 0].pt.tolist()
    df['leptonEta'] = leptons[:, 0].eta.tolist()
    df['leptonPhi'] = leptons[:, 0].phi.tolist()

    # Transverse top mass
    delta_phi = np.abs((bJets[:, 0] + leptons[:, 0]).phi - events.MET.fiducialGenPhi)
    topMt = np.sqrt(2 * (bJets[:, 0] + leptons[:, 0]).pt * events.MET.fiducialGenPt * (1 - np.cos(delta_phi)))
    df['topMt'] = topMt.tolist()

    # Delta y between leading b-jet and leading jet
    bJetLead = ak.firsts(
        genJets[(np.abs(genJets.partonFlavour) == 5) & (np.abs(genJets.eta) < 2.5)]
    )
    jetLead = ak.firsts(genJets[~(genJets.pt == bJetLead.pt)])

    df['bJetPt'] = bJetLead.pt.tolist()
    df['bJetEta'] = bJetLead.eta.tolist()
    df['bJetPhi'] = bJetLead.phi.tolist()

    df['jetLeadPt'] = jetLead.pt.tolist()
    df['jetLeadEta'] = jetLead.eta.tolist()
    df['jetLeadPhi'] = jetLead.phi.tolist()

    df['delta_eta_b_j'] = np.abs(jetLead.eta - bJetLead.eta).tolist()
    df['rapiditySumJH'] = np.sqrt(jetLead.rapidity**2 + diphoton.rapidity**2).tolist()
    df['rapidityDiffJH'] = np.abs(jetLead.rapidity - diphoton.rapidity).tolist()
    df['DeltaY_Hb'] = np.abs(bJetLead.rapidity - diphoton.rapidity).tolist()
    df['DeltaY_Hl'] = np.abs(leptons[:, 0].rapidity - diphoton.rapidity).tolist()

    df["weights"] = weights

    # Display the DataFrame to verify
    print(df.head())

    return df




basepath = "/net/scratch_cms3a/mausolf/HttCPAnalysis/NanoGenStudies/"
filenamesTtHCPEven = glob.glob(basepath + "sampleGeneration/ttHCPEven/*.root")
filenamesTHqCPEven = glob.glob(basepath + "sampleGeneration/tHq_TLep_k1_ktilde0/*.root")


filenamesTtHCPEven = [f for f in filenamesTtHCPEven if not "_5.root" in f]

stop = None
# stop = 300

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
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

    # Second split: validation set and test set
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save the DataFrames to CSV files
    train_df.to_csv(os.path.join(output_dir, f'{name}_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, f'{name}_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, f'{name}_test.csv'), index=False)

    print(f"{name} split into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
