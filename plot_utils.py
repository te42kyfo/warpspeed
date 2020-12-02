#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import math
import random


def volumeScatterPlot(measurements, predictions, title=None):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)
    markers = [".", "o", "*", "s", "P", "D", "x", "+", "+", "+", "+", "+", "+"]

    for r in range(-1, 11):
        keys = [bc[1:] for bc in measurements.keys() if bc[0] == r]
        if len(keys) == 0:
            continue

        random.shuffle(keys)

        measuredValues = [measurements[(r, *key)] for key in keys]
        predictedValues = [predictions[(r, *key)] for key in keys]

        colors = [tuple(math.log2(bc) for bc in b) for b in keys]

        maxColors = [max([color[i] for color in colors]) for i in range(3)]
        maxSize = max([(color[0] + color[1] + color[2]) for color in colors])
        colors = [
            tuple(
                color[i]
                / max(color)
                * ((color[0] + color[1] + color[2]) / maxSize * 0.1 + 0.9)
                for i in range(3)
            )
            for color in colors
        ]

        ax.scatter(
            measuredValues,
            predictedValues,
            s=[300] * len(colors),
            c=colors,
            alpha=0.05,
            edgecolors="none",
        )
        ax.scatter(
            measuredValues,
            predictedValues,
            s=[30] * len(colors),
            c=colors,
            marker=markers[r + 1],
            alpha=1,
            edgecolors="none",
        )

    ax.set_xscale("log", subsx=[])
    ax.set_yscale("log", subsy=[])

    # ax.set_xlim((7.5,30))
    # ax.set_ylim((7.5,30))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot([7, 1150], [7, 1150], color="black", alpha=0.2)

    ax.set_xticks(
        [0.1, 1, 2, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    )
    ax.set_yticks(
        [0.1, 1, 2, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    )

    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x))
    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)

    lim = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))
    ax.set_xlim(
        (max(min(lim[0] * 0.8, lim[0] - 0.1), 0.1), max(1, lim[1] * 1.2, lim[1] + 0.1))
    )
    ax.set_ylim(
        (max(min(lim[0] * 0.8, lim[0] - 0.1), 0.1), max(1, lim[1] * 1.2, lim[1] + 0.1))
    )

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel("measured Volume, B/Lup")
    ax.set_ylabel("predicted Volume, B/Lup")

    fig.tight_layout()
    ax.grid()
    plt.show()


def performanceScatterPlot(measuredValues, predictedValues):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)

    colors = [
        tuple(math.log2(bc) / math.log2(1024) for bc in b)
        for b in measuredValues.keys()
    ]

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
    # ax.scatter(list(measuredValues.values()), list(predictedValues.values()), ".", markersize=1)

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
    plt.show()
