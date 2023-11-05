#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
import math
import random

markers = ["o", "s", "D", "P", "X", "h", "*", "v", "^", "<", ">"]


def computeMape(values):
    mape = 0
    for v in values:
        mape += abs(v[1] - v[2]) / v[1]
    return mape / len(values)


def getColor(b):
    return tuple(
        min(
            1.0,
            math.log2(c) / math.log2(128) * 1.5 / math.log2(b[0] * b[1] * b[2]) * 10,
        )
        for c in b
    )


def volumeScatterPlot(values, title=None, lims=None, linear=False):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)

    if len(values[0]) < 4:
        values = [(*v, 0) for v in values]

    categories = sorted(list(set([v[3] for v in values])))

    for catID, catName in enumerate(categories):
        catValues = [v for v in values if v[3] == catName]
        random.shuffle(catValues)

        measValues = [v[1] for v in catValues]
        predValues = [v[2] for v in catValues]

        colors = [getColor(v[0]) for v in catValues]
        ax.scatter(
            measValues,
            predValues,
            s=[600] * len(colors),
            marker="o",
            c=colors,
            alpha=0.03,
            edgecolors="none",
        )
        ax.scatter(
            measValues,
            predValues,
            s=[50] * len(colors),
            c=colors,
            marker=markers[catID],
            alpha=1,
            edgecolors="none",
            label=catName,
        )

        if len(catValues[0]) > 4:
            selectedValues = [v for v in catValues if abs(v[2] - v[4]) / v[2] > 0.1]
            selectedMeas = [v[1] for v in selectedValues]
            selectedPred = [v[4] for v in selectedValues]

            ax.scatter(
                selectedMeas,
                selectedPred,
                s=5,
                color="none",
                marker=markers[catID],
                alpha=0.5,
                edgecolors="black",
                zorder=-1,
            )
            for v in selectedValues:
                ax.arrow(
                    v[1],
                    v[4],
                    0,
                    v[2] - v[4],
                    linewidth=0.5,
                    alpha=0.5,
                    head_width=0.05 * v[1],
                    fill=False,
                    head_length=0.05 * v[4],
                    length_includes_head=True,
                    color="gray",
                )
    plot_range = (
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    )
    if not linear:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xticks(
        [
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            1.0,
            2,
            4,
            8,
            9,
            12,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
        ]
    )
    ax.set_yticks(
        [
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            1.0,
            2,
            4,
            8,
            9,
            12,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
        ]
    )

    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x))
    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)

    if not lims is None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    else:
        lim = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))
        ax.set_xlim(
            (
                max(min(lim[0] * 0.9, lim[0] - 0.02), 0.001),
                max(0.5, lim[1] * 1.1, lim[1] + 0.02),
            )
        )
        ax.set_ylim(
            (
                max(min(lim[0] * 0.9, lim[0] - 0.02), 0.001),
                max(0.5, lim[1] * 1.1, lim[1] + 0.02),
            )
        )

    ax.plot(
        ax.get_xlim(),
        ax.get_ylim(),
        color="black",
        alpha=0.2,
    )

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel("actual Volume, B/Lup")
    ax.set_ylabel("predicted Volume, B/Lup")

    tau, p_value = stats.kendalltau([v[1] for v in values], [v[2] for v in values])

    text = "MAPE: {:.1f}%,  Kendall's Tau: {:.2f}".format(
        computeMape(values) * 100, tau
    )
    # text = "MAPE: {:.1f}%".format(computeMape(values) * 100)
    ax.annotate(
        text,
        (0.04, 0.95),
        xycoords="axes fraction",
        va="center",
        ha="left",
        fontfamily="monospace",
        fontsize="small",
        bbox=dict(boxstyle="round", fc="#FFFFFFAA", ec="#CCCCCC00"),
    )

    ax.legend(loc="lower right")
    fig.tight_layout()
    ax.grid()

    plt.savefig("./autoplots/" + title + ".svg")
    # plt.show()
    # plt.close()

    return (fig, ax)


def performanceScatterPlot(measuredValues, predictedValues):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)

    colors = [tuple(getColor(bc) for bc in b) for b in measuredValues.keys()]

    ax.scatter(
        measuredValues.values(),
        predictedValues.values(),
        s=[200] * len(colors),
        c=colors,
        alpha=0.01,
        edgecolors="none",
    )
    ax.scatter(
        measuredValues.values(),
        predictedValues.values(),
        s=[10] * len(colors),
        c=colors,
        alpha=1,
        edgecolors="none",
    )

    fig.tight_layout()

    ax.set_xscale("log", subsx=[])
    ax.set_yscale("log", subsy=[])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot([0.01, 10], [0.01, 10], color="black", alpha=0.2)

    ax.set_xticks([0.5, 1.0, 2.0, 4.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([0.5, 1.0, 2.0, 4.0])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlim((min(xlim[0], ylim[0]), max(xlim[1], ylim[1])))
    ax.set_ylim((min(xlim[0], ylim[0]), max(xlim[1], ylim[1])))

    ax.set_title("Performance")
    ax.set_xlabel("measured Performance, GLup/s")
    ax.set_ylabel("predicted Performance, GLup/s")

    ax.grid()
    plt.tight_layout()
    plt.savefig("./autoplots/" + title + ".svg")
    # plt.show()
    return (fig, ax)


def scatterPlot(values, title=None, lims=None):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)

    if len(values[0]) < 4:
        values = [(*v, 0) for v in values]

    categories = sorted(list(set([v[3] for v in values])))

    for catID, catName in enumerate(categories):
        catValues = [v for v in values if v[3] == catName]
        random.shuffle(catValues)

        measValues = [v[1] for v in catValues]
        predValues = [v[2] for v in catValues]

        colors = [getColor(v[0]) for v in catValues]
        ax.scatter(
            measValues,
            predValues,
            s=[400] * len(colors),
            marker="o",
            c=colors,
            alpha=0.05,
            edgecolors="none",
        )
        ax.scatter(
            measValues,
            predValues,
            s=[20] * len(colors),
            c=colors,
            marker=markers[catID],
            alpha=1,
            edgecolors="none",
            label=catName,
        )

        if len(catValues[0]) > 4:
            selectedValues = [v for v in catValues if abs(v[2] - v[4]) / v[2] > 0.05]
            selectedMeas = [v[1] for v in selectedValues]
            selectedPred = [v[4] for v in selectedValues]

            ax.scatter(
                selectedMeas,
                selectedPred,
                s=30,
                color="none",
                marker=markers[catID],
                alpha=1,
                edgecolors="gray",
            )
            for v in selectedValues:
                ax.arrow(
                    v[1],
                    v[4],
                    0,
                    v[2] - v[4],
                    linewidth=0.5,
                    head_width=0.05 * v[1],
                    fill=False,
                    head_length=0.05 * v[4],
                    length_includes_head=True,
                    color="gray",
                )
    plot_range = (
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    )

    if not lims is None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel("Quantity A")
    ax.set_ylabel("Quantity B")
    ax.legend()
    fig.tight_layout()
    ax.grid()

    plt.savefig("./autoplots/" + title + ".svg")
    plt.show()
    plt.close()
    return (fig, ax)
