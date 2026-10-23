import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, DelphesSchema
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import sys
import glob
sys.path.append("./..")
from modules import plotter
import vector
plt.style.use([hep.style.CMS])
vector.register_awkward()


def event_selection(events, channel="lep"):

    # https://github.com/cms-sw/cmssw/blob/master/GeneratorInterface/RivetInterface/python/particleLevel_cfi.py

    photons = events.Photon
    photons = photons[
        (photons.pt > 25) &
        (np.abs(photons.eta) < 2.5)
    ]
    ### string vs. dot notation, use string notation to assign values!
    events["Photon"] = photons

    #### lepton selections
    electrons = events.Electron

    electrons = electrons[
        (np.abs(electrons.eta) < 2.4) & (electrons.pt > 10)
    ]

    electrons["mass"] = 511e-6 * np.ones(len(electrons))
    events["Electron"] = electrons

    muons = events.Muon

    muons = muons[
        (np.abs(muons.eta) < 2.4) & (muons.pt > 10)
    ]

    muons["mass"] = 0.106 * np.ones(len(muons))
    events["Muon"] = muons

    # Concatenate electrons and muons
    leptons = ak.concatenate([electrons, muons], axis=1)

    leptons["pt"] = leptons.PT
    leptons["eta"] = leptons.Eta
    leptons["phi"] = leptons.Phi

    # Sort the concatenated collection by pt
    leptons = leptons[ak.argsort(leptons.pt, ascending=False)]

    # Assign the sorted collection to a new field "Lepton" in the events
    events["Lepton"] = leptons

    # jet selections
    jets = events.Jet
    events["Jet"] = jets[
        (jets.pt > 25) & (np.abs(jets.eta) < 4.7) 
        & ak.all(jets.metric_table(events.Photon, metric=lambda a, b: a.delta_r(b)) > 0.5, axis=-1)
        & ak.all(jets.metric_table(events.Lepton, metric=lambda a, b: a.delta_r(b)) > 0.5, axis=-1)
    ]

    events = events[
        (ak.num(events.Photon) == 2)
        & ak.fill_none(
            np.max(events.Photon.pt, axis=1) > 35., False
        )
        & (ak.num(events.Jet) >= 2) 
        # require one central b-jet
        & ak.any(
            events.Jet[np.abs(events.Jet.eta) < 2.5].BTag == 1, axis=1
        )
    ]

    ### MGG fiducial cuts:
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

    diphoton_mass = diphoton.mass

    events = events[
        (phoLead.pt / diphoton_mass > 1/3)
        & (phoSublead.pt / diphoton_mass > 1/4)
        & (diphoton_mass > 100)
        & (diphoton_mass < 180)
    ]

    if channel == "lep":
        events = events[ak.num(events.Lepton) > 0]
    elif channel == "had":
        events = events[ak.num(events.Lepton) == 0]
    return events

basepath = "/net/scratch_cms3a/mausolf/HttCPAnalysis/NanoGenStudies/"

# Function to decode flag values
def decode_flags(flag_value):
    # Decoding the flags based on bit positions
    flag_descriptions = [
        "isPrompt",
        "isDecayedLeptonHadron",
        "isTauDecayProduct",
        "isPromptTauDecayProduct",
        "isDirectTauDecayProduct",
        "isDirectPromptTauDecayProduct",
        "isDirectHadronDecayProduct",
        "isHardProcess",
        "fromHardProcess",
        "isHardProcessTauDecayProduct",
        "isDirectHardProcessTauDecayProduct",
        "fromHardProcessBeforeFSR",
        "isFirstCopy",
        "isLastCopy",
        "isLastCopyBeforeFSR"
    ]
    binary = bin(flag_value)[2:].zfill(15)  # Binary representation of flag value
    set_flags = [desc for bit, desc in zip(reversed(binary), flag_descriptions) if bit == '1']
    return set_flags


def main(path, hists, entryStop=None):

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
    }, with_name="Momentum4D")

    bJets = jets[(jets.BTag == 1) & (np.abs(jets.eta) < 2.5)]

    weights = events.Weight.Weight[:,0]

    hists["mass"][0].fill(diphoton.mass, weight=weights)
    hists["ptH"][0].fill(diphoton.pt, weight=weights)
    hists["yH"][0].fill(diphoton.rapidity, weight=weights)

    hists["leadingPhotonPt"][0].fill(phoLead.pt, weight=weights)
    hists["leadingPhotonEta"][0].fill(phoLead.eta, weight=weights)

    hists["subleadingPhotonPt"][0].fill(phoSublead.pt, weight=weights)
    hists["subleadingPhotonEta"][0].fill(phoSublead.eta, weight=weights)

    hists["nJets"][0].fill(ak.num(events.Jet), weight=weights)
    hists["nLeptons"][0].fill(ak.num(events.Lepton), weight=weights)

    hists["nJetsCentral"][0].fill(
        ak.num(events.Jet[np.abs(events.Jet.eta)<2.5]), weight=weights
    )
    hists["nBjetsCentral"][0].fill(
        ak.num(
            events.Jet[
                (np.abs(events.Jet.eta)<2.5) & (events.Jet.BTag == 1)
            ]
        ),
        weight=weights
    )
    hists["nJetsForward"][0].fill(
        ak.num(events.Jet[np.abs(events.Jet.eta)>2.5]), weight=weights
    )

    hists["bJetsEta"][0].fill(
        np.abs(ak.flatten(bJets.eta)), 
        weight=ak.flatten(ak.broadcast_arrays(weights, bJets.eta)[0])
    )

    hists["met"][0].fill(events.MissingET.MET, weight=weights)

    hists["leptonPt"][0].fill(
        ak.flatten(events.Lepton.pt),
        weight=ak.flatten(ak.broadcast_arrays(weights, events.Lepton.pt)[0])
    )

    # Calculate the transverse top mass
    delta_phi = (bJets[:, 0] + leptons[:, 0]).phi - events.MissingET.Phi
    topMt = np.sqrt(2 * (bJets[:, 0] + leptons[:, 0]).pt * events.MissingET.MET * (1 - np.cos(delta_phi)))

    hists["topMt"][0].fill(topMt, weight=weights)

    ##### get delta_y (leading b-jet, leading jet)
    bJetLead = ak.firsts(
        bJets
    )

    # identify by requiring same pt, might be an issue if two jets in an event have same pt...
    jetLead = ak.firsts(jets[~(jets.pt == bJetLead.pt)])

    hists["leading_bJetPt"][0].fill(bJetLead.pt, weight=weights) 
    hists["leading_bJetEta"][0].fill(bJetLead.eta, weight=weights) 

    hists["leadingJetPt"][0].fill(jetLead.pt, weight=weights)
    hists["leadingJetEta"][0].fill(jetLead.eta, weight=weights)

    hists["delta_eta_b_j"][0].fill((jetLead.eta - bJetLead.eta), weight=weights)

    hists["rapiditySumJH"][0].fill(
        np.sqrt(jetLead.rapidity**2 + diphoton.rapidity**2), 
        weight=weights
    )

    hists["rapidityDiffJH"][0].fill(
        (jetLead.rapidity - diphoton.rapidity), 
        weight=weights
    )

    hists["DeltaY_Hb"][0].fill(
        (bJetLead.rapidity - diphoton.rapidity), 
        weight=weights
    )

    hists["DeltaY_Hl"][0].fill(
        (leptons[:, 0].rapidity - diphoton.rapidity), 
        weight=weights
    )

    return hists

if __name__ == "__main__":

    filenames_ttH = glob.glob("/net/scratch_cms3a/jerdmann/tag_1_pythia8_events.root")
    filenames_tHq = glob.glob(basepath + "sampleGeneration/tHq_TLep_k1_ktilde0/*.root")

    hists_ttH = plotter.get_dict_hists()
    hists_tHq = plotter.get_dict_hists()

    stop = None 
    for file in filenames_ttH:
        hists_ttH = main(file, hists_ttH, entryStop=stop)
    for file in filenames_tHq:
        hists_tHq = main(file, hists_tHq, entryStop=stop)

    histNames = [
        "mass",
        "ptH",
        "yH",
        "nJets",
        "nBjetsCentral",
        "nJetsCentral",
        "nJetsForward",
        "leadingJetPt",
        "leadingJetEta",
        "leadingPhotonPt",
        "leadingPhotonEta",
        "subleadingPhotonPt",
        "subleadingPhotonEta",
        "leading_bJetPt",
        "leading_bJetEta",
        "delta_eta_b_j",
        "nLeptons",
        "leptonPt",
        "met",
        "topMt",
        "rapiditySumJH",
        "rapidityDiffJH",
        "DeltaY_Hb",
        "DeltaY_Hl"
    ]

    for histName in histNames:

        plotter.plot_histograms(
            hists=[hists_ttH[histName][0], hists_tHq[histName][0]],
            process_labels=[r"$t\bar{t}H$", r"$tH$"],
            output_filename=f"./PlotsDelphes/{histName}.pdf",
            normalize=True,
            axis_labels=hists_ttH[histName][1],
            include_flow=True
        )
