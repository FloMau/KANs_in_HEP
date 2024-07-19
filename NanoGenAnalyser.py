import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
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

    # print(events.fields)

    genIsoPhotons = events.GenIsolatedPhoton

    genIsoPhotons = genIsoPhotons[
        (genIsoPhotons.pt > 25) &
        # (np.abs(genIsoPhotons.eta) < 1.4442) | ((np.abs(genIsoPhotons.eta) < 2.5) & (np.abs(genIsoPhotons.eta) > 1.566))
        (np.abs(genIsoPhotons.eta) < 2.5)
    ]

    ### string vs. dot notation, use string notation to assign values!
    events["GenIsolatedPhoton"] = genIsoPhotons

    #### lepton selections
    genDressedLeptons = events.GenDressedLepton

    genDressedLeptons = genDressedLeptons[
        ((np.abs(genDressedLeptons.pdgId) == 11) | (np.abs(genDressedLeptons.pdgId) == 13)) &
        (np.abs(genDressedLeptons.eta < 2.4)) & (genDressedLeptons.pt > 10)
    ]
    events["GenDressedLepton"] = genDressedLeptons

    # jet selections
    genJets = events.GenJet

    events["GenJet"] = genJets[
        (genJets.pt > 25) & (np.abs(genJets.eta) < 4.7) 
        & ak.all(genJets.metric_table(events.GenIsolatedPhoton, metric=lambda a, b: a.delta_r(b)) > 0.4, axis=-1)
        & ak.all(genJets.metric_table(events.GenDressedLepton, metric=lambda a, b: a.delta_r(b)) > 0.4, axis=-1)
        & (genJets.partonFlavour != 0)
    ]
    # print(len(events), len((np.max(events.GenIsolatedPhoton.pt, axis=1) > 35.)))
    events = events[
        (ak.num(events.GenIsolatedPhoton) == 2)
        & ak.fill_none(
            np.max(events.GenIsolatedPhoton.pt, axis=1) > 35., False
        )
        & (ak.num(events.GenJet) >= 2) 
        # require one central b-jet
        & (ak.num(
            events.GenJet[(np.abs(events.GenJet.eta) < 2.5) & (np.abs(events.GenJet.partonFlavour) == 5)]
        ) > 0)
    ]

    if channel == "lep":
        events = events[ak.num(events.GenDressedLepton) > 0]
    elif channel == "had":
        events = events[ak.num(events.GenDressedLepton) == 0]
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
    events = NanoEventsFactory.from_root(path, schemaclass=NanoAODSchema, entry_stop=entryStop).events()
    nEvents = len(events)
    events = event_selection(events)

    print("\tINFO: number of events before & after selection:", nEvents, len(events))

    # status: https://pythia.org/latest-manual/ParticleProperties.html
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

    hists["ptH"][0].fill(diphoton.pt, weight=weights)
    hists["yH"][0].fill(diphoton.rapidity, weight=weights)

    hists["leadingPhotonPt"][0].fill(phoLead.pt, weight=weights)
    hists["leadingPhotonEta"][0].fill(phoLead.eta, weight=weights)

    hists["subleadingPhotonPt"][0].fill(phoSublead.pt, weight=weights)
    hists["subleadingPhotonEta"][0].fill(phoSublead.eta, weight=weights)

    hists["nJets"][0].fill(ak.num(events.GenJet), weight=weights)
    hists["nLeptons"][0].fill(ak.num(events.GenDressedLepton), weight=weights)

    hists["nJetsCentral"][0].fill(
        ak.num(events.GenJet[np.abs(events.GenJet.eta)<2.5]), weight=weights
    )
    hists["nBjetsCentral"][0].fill(
        ak.num(events.GenJet[(np.abs(events.GenJet.eta)<2.5) & (np.abs(events.GenJet.partonFlavour)==5)]), weight=weights
    )
    hists["nJetsForward"][0].fill(
        ak.num(events.GenJet[np.abs(events.GenJet.eta)>2.5]), weight=weights
    )

    hists["bJetsEta"][0].fill(
        np.abs(ak.flatten(bJets.eta)), 
        weight=ak.flatten(ak.broadcast_arrays(weights, bJets.eta)[0])
    )

    hists["met"][0].fill(events.MET.fiducialGenPt, weight=weights)

    hists["leptonPt"][0].fill(
        ak.flatten(events.GenDressedLepton.pt),
        weight=ak.flatten(ak.broadcast_arrays(weights, events.GenDressedLepton.pt)[0])
    )

    # Calculate the transverse top mass
    delta_phi = np.abs((bJets[:, 0] + leptons[:, 0]).phi - events.MET.fiducialGenPhi)
    topMt = np.sqrt(2 * (bJets[:, 0] + leptons[:, 0]).pt * events.MET.fiducialGenPt * (1 - np.cos(delta_phi)))

    hists["topMt"][0].fill(topMt, weight=weights)

    ##### get delta_y (leading b-jet, leading jet)
    ## assuming they are sorted by pt, needs to be changed eventually
    bJetLead = ak.firsts(
        genJets[(np.abs(genJets.partonFlavour) == 5) & (np.abs(genJets.eta) < 2.5)]
    )
    # identify by requiring same pt, might be an issue if two jets in an event have same pt...
    jetLead = ak.firsts(genJets[~(genJets.pt == bJetLead.pt)])

    hists["leading_bJetPt"][0].fill(bJetLead.pt, weight=weights) 
    hists["leading_bJetEta"][0].fill(bJetLead.eta, weight=weights) 

    hists["leadingJetPt"][0].fill(jetLead.pt, weight=weights)
    hists["leadingJetEta"][0].fill(jetLead.eta, weight=weights)

    hists["delta_eta_b_j"][0].fill(np.abs(jetLead.eta - bJetLead.eta), weight=weights)

    hists["rapiditySumJH"][0].fill(
        np.sqrt(jetLead.rapidity**2 + diphoton.rapidity**2), 
        weight=weights
    )

    hists["rapidityDiffJH"][0].fill(
        np.abs(jetLead.rapidity - diphoton.rapidity), 
        weight=weights
    )

    hists["DeltaY_Hb"][0].fill(
        np.abs(bJetLead.rapidity - diphoton.rapidity), 
        weight=weights
    )

    hists["DeltaY_Hl"][0].fill(
        np.abs(leptons[:, 0].rapidity - diphoton.rapidity), 
        weight=weights
    )

    return hists

if __name__ == "__main__":

    filenamesTtHCPEven = glob.glob(basepath + "sampleGeneration/ttHCPEven/*.root")
    filenamesTtHCPEven = glob.glob("./sampleGeneration/ttH/*.root")
    filenamesTtHCPOdd = glob.glob(basepath + "sampleGeneration/ttHCPOdd/*.root")
    filenamesTHqCPEven = glob.glob(basepath + "sampleGeneration/tHq_TLep_k1_ktilde0/*.root")
    filenamesTHqCPEven5FS = glob.glob(basepath + "sampleGeneration/tHq_TLep_k1_ktilde0_5FS/*.root")
    filenamesTHqCPOdd = glob.glob(basepath + "sampleGeneration/tHq_TLep_k0_ktilde1/*.root")
    filenamesTHqCPOddCrossCheck = glob.glob(basepath + "sampleGeneration/tHq_TLep_k0_ktilde1_cross_check/*.root")

    filenamesTtHCPEven = [f for f in filenamesTtHCPEven if not "_5.root" in f]

    histsTtHCPEven = plotter.get_dict_hists()
    histsTtHCPOdd = plotter.get_dict_hists()
    histsTHqCPEven = plotter.get_dict_hists()
    histsTHqCPOdd = plotter.get_dict_hists()
    histsTHqCPEven5FS = plotter.get_dict_hists()
    histsTHqCPOddCrossCheck = plotter.get_dict_hists()

    stop = None #
    # stop = 10_000
    # stop = 300
    for file in filenamesTtHCPEven:
        histsTtHCPEven = main(file, histsTtHCPEven, entryStop=stop)
    # for file in filenamesTtHCPOdd:
    #     histsTtHCPOdd = main(file, histsTtHCPOdd, entryStop=stop)
    for file in filenamesTHqCPEven:
        histsTHqCPEven = main(file, histsTHqCPEven, entryStop=stop)
    # for file in filenamesTHqCPOdd:
    #     histsTHqCPOdd = main(file, histsTHqCPOdd, entryStop=stop)
    # for file in filenamesTHqCPEven5FS:
    #     histsTHqCPEven5FS = main(file, histsTHqCPEven5FS, entryStop=stop)
    # for file in filenamesTHqCPOddCrossCheck:
    # for file in filenamesTHqCPOddCrossCheck:
    #     histsTHqCPOddCrossCheck = main(file, histsTHqCPOddCrossCheck, entryStop=stop)


    labels = ["ttH CP even", "ttH CP odd", "tH CP even", "tH CP odd"]

    histNames = [
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

        # plotter.plot_histograms(
        #     hists=[histsTtHCPEven[histName][0], histsTtHCPOdd[histName][0], histsTHqCPEven[histName][0], histsTHqCPOdd[histName][0]],
        #     process_labels=labels,
        #     output_filename=f"./Plots/{histName}.pdf",
        #     normalize=True,
        #     axis_labels=histsTtHCPEven[histName][1],
        #     include_flow=True
        # )

        # plotter.plot_histograms(
        #     hists=[histsTHqCPEven[histName][0], histsTHqCPEven5FS[histName][0]],
        #     process_labels=["tHq 4FS", "tHq 5FS"],
        #     output_filename=f"./PlotsFS/{histName}.pdf",
        #     normalize=True,
        #     axis_labels=histsTtHCPEven[histName][1],
        #     include_flow=True
        # )

        # plotter.plot_histograms(
        #     hists=[histsTHqCPOdd[histName][0], histsTHqCPOddCrossCheck[histName][0]],
        #     process_labels=["tHq CP-odd", "Cross-check"],
        #     output_filename=f"./PlotsCrossCheck/{histName}.pdf",
        #     normalize=True,
        #     axis_labels=histsTtHCPEven[histName][1],
        #     include_flow=True
        # )

        plotter.plot_histograms(
            hists=[histsTtHCPEven[histName][0], histsTHqCPEven[histName][0]],
            process_labels=[r"$t\bar{t}H$", r"$tH$"],
            output_filename=f"./PlotsInputFeatures/{histName}.pdf",
            normalize=True,
            axis_labels=histsTtHCPEven[histName][1],
            include_flow=True
        )
