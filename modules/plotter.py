import matplotlib.pyplot as plt 
import mplhep as hep
plt.style.use([hep.style.ROOT])
import os
import numpy as np
from hist import Hist
from copy import deepcopy


def plot_histograms(
    hists,
    process_labels,
    output_filename, 
    axis_labels=("x-axis","Events"), 
    normalize=False,
    linestyle="solid",
    log=False,
    include_flow=False,
    data=False,
    lumi=None,
    return_figure=False
):

    if include_flow:
        hists = [include_overflow_underflow(hist) for hist in hists]
    if normalize:
        integrals = [_hist.sum().value for _hist in hists]
        hists = [_hist / integral for _hist, integral in zip(hists, integrals)]
    binning = hists[0].to_numpy()[1]
    values = [_hist.values() for _hist in hists]
    uncertainties = [np.sqrt(_hist.variances()) for _hist in hists]

    fig, ax = plt.subplots(figsize=(10,9))
    hep.histplot(
        values,
        label=process_labels,
        bins=binning,
        linewidth=3,
        yerr=uncertainties,
        ax=ax,
        linestyle=linestyle
    )

    ax.margins(y=0.15)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.05*ax.get_ylim()[1])
        ax.tick_params(labelsize=22)
    ax.set_xlabel(axis_labels[0], fontsize=26)
    ax.set_ylabel(axis_labels[1], fontsize=26)
    ax.tick_params(labelsize=24)

    # adjust legend line width
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
        handle.get_children()[0].set_linewidth(3)  # Line
        handle.get_children()[1].set_linewidth(3)  # Caplines
        # handle.get_children()[2].set_linewidth(3)  # Bars
        new_handles.append(handle)
    ncols = 1 if len(hists) < 4 else 2
    ax.legend(handles=new_handles, labels=labels, loc="upper right", fontsize=24, ncols=ncols)#, handlelength=3)

    # in case the figure should be modified later on, use return_figure=True
    if not return_figure:
        if not os.path.exists(output_filename.replace(output_filename.split("/")[-1], "")):
            os.makedirs(output_filename.replace(output_filename.split("/")[-1], ""))
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close()
        return
    else:
        return fig, ax



def include_overflow_underflow(hist):

    # do not change in place:
    new_hist = deepcopy(hist)
    bin_contents = new_hist.view(flow=True)
    overflow, underflow = bin_contents[-1], bin_contents[0]
    n_bins = len(bin_contents) - 2 
    # print(new_hist.view(flow=True))
    new_hist[1] += underflow
    new_hist[n_bins-1] += overflow
    # print(new_hist.view(flow=True))
    return new_hist


def get_dict_hists():

    dict_histograms = {
        "mass": [Hist.new.Reg(50, 100, 150).Weight(), (r"Diphoton mass [GeV]", "Fraction of events")],
        "topMt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Transverse top quark mass [GeV]", "Fraction of events")],
        "ptH": [Hist.new.Reg(20, 0, 400).Weight(), (r"Diphoton $p_T$ [GeV]", "Fraction of events")],
        "yH": [Hist.new.Reg(20, -3, 3).Weight(), (r"Diphoton $y$", "Fraction of events")],

        "nJets": [Hist.new.Reg(13, -0.5, 12.5).Weight(), (r"Number of gen jets", "Fraction of events")],
        "nJetsCentral": [Hist.new.Reg(11, -0.5, 10.5).Weight(), (r"Number of central jets", "Fraction of events")],
        "nJetsForward": [Hist.new.Reg(5, -0.5, 4.5).Weight(), (r"Number of forward jets", "Fraction of events")],
        "nBjetsCentral": [Hist.new.Reg(5, -0.5, 4.5).Weight(), (r"Number of central b-jets", "Fraction of events")],
        "nBtags": [Hist.new.Reg(5, -0.5, 4.5).Weight(), (r"Number of b-tagged jets", "Fraction of events")],

        "bJetsEta": [Hist.new.Reg(20, 0., 4.7).Weight(), (r"b-jets $\eta$", "Fraction of events")],

        "leading_bJetPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"b-jet $p_T$ [GeV]", "Fraction of events")],
        "leading_bJetEta": [Hist.new.Reg(20, -4.7, 4.7).Weight(), (r"b-jet $\eta$", "Fraction of events")],

        "leadingJetPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Leading jet $p_T$ [GeV]", "Fraction of events")],
        "leadingJetEta": [Hist.new.Reg(20, -4.7, 4.7).Weight(), (r"Leading jet $\eta$", "Fraction of events")],

        "subleadingJetPt": [Hist.new.Reg(20, 0, 200).Weight(), (r"Subleading jet $p_T$ [GeV]", "Fraction of events")],
        "subleadingJetEta": [Hist.new.Reg(20, -4.7, 4.7).Weight(), (r"Subleading jet $\eta$", "Fraction of events")], 

        "nLeptons": [Hist.new.Reg(4, -0.5, 3.5).Weight(), (r"Number of leptons", "Fraction of events")],
        "leptonPt": [Hist.new.Reg(20, 0, 200).Weight(), (r"Lepton $p_T$ [GeV]", "Fraction of events")],
        "leptonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Lepton $\eta$", "Fraction of events")],
        
        "leadingPhotonPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Leading photon $p_T$ [GeV]", "Fraction of events")],
        "leadingPhotonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Leading photon $\eta$", "Fraction of events")],
        "subleadingPhotonPt": [Hist.new.Reg(20, 0, 200).Weight(), (r"Subleading photon $p_T$ [GeV]", "Fraction of events")],
        "subleadingPhotonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Subleading photon $\eta$", "Fraction of events")],

        "met": [Hist.new.Reg(20, 0, 200).Weight(), (r"MET $p_T$ [GeV]", "Fraction of events")],

        "rapiditySumJH": [Hist.new.Reg(15, 0, 6).Weight(), (r"$y^{j} \oplus y^{\gamma\gamma}$", "Fraction of events")],
        "rapidityDiffJH": [Hist.new.Reg(15, 0, 6).Weight(), (r"$|\Delta y(j, \gamma\gamma)|$", "Fraction of events")],
        "DeltaY_Hb": [Hist.new.Reg(15, 0, 5).Weight(), (r"$|\Delta y(\gamma\gamma, b)|$", "Fraction of events")],
        "DeltaY_Hl": [Hist.new.Reg(15, 0, 5).Weight(), (r"$|\Delta y(\gamma\gamma, l)|$", "Fraction of events")],
        "delta_eta_b_j": [Hist.new.Reg(20, 0, 7).Weight(), (r"$\Delta \eta(b,j)$", "Fraction of events")],
        "delta_r_jet_pho": [Hist.new.Reg(60, 0, 8).Weight(), (r"$\Delta R(j, \gamma)$", "Fraction of events")]
    }

    return dict_histograms
