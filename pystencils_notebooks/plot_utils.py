#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random


def volumeScatterPlot(measurements, predictions, title=None, lims=None):
    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(150)
    markers = [
        #".",
        "D",
        "s",
        "s",
        "o",
        "v",
        "s",
        "1",
        "2",
        "3",
        "4",
        "v",
        "^",
        "<",
        ">",
    ]

    labels = ["", "DRAM", "L2", "L1", ""]

    for r in range(-1, 13):
        keys = [
            bc[1:]
            for bc in measurements.keys()
            if bc[0] == r
            if predictions[bc] > 0.1 and measurements[bc] > 0.1
        ]
        if len(keys) == 0:
            continue
        random.shuffle(keys)

        measuredValues = [measurements[(r, *key)] for key in keys]
        predictedValues = [predictions[(r, *key)] for key in keys]

        colors = [tuple(math.log2(bc) for bc in b) for b in keys]

        maxColors = [max([color[i] for color in colors]) for i in range(len(colors[0]))]
        maxSize = max([sum(color) for color in colors])
        colors = [
            tuple(
                [
                    min(1.0, color[i] / math.log2(128) * 1.5 )
                    for i in range(len(colors[0]))
                ]
                + [0] * (3 - len(colors[0]))
            )
            for color in colors
        ]

        colorArray = [(1.0, 0.0, 0.0), (1.0, 0.9, 0.0), (0.0, 1.0, 0.0), (0.0, 0.9, 0.7), (0.0, 0.0, 1.0), (0.7, 0.0, 0.7)]

        #colors = len(colors) * [colorArray[r-1]]
        ax.scatter(
            measuredValues,
            predictedValues,
            s=[600] * len(colors),
            marker="o",
            c=colors,
            alpha=0.05,
            edgecolors="none",
        )
        ax.scatter(
            measuredValues,
            predictedValues,
            s=[40] * len(colors),
            c=colors,
            marker=markers[r + 1],
            alpha=1,
            edgecolors="none",
            label=labels[r],
        )

        plot_range = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
                      max(ax.get_xlim()[1], ax.get_ylim()[1]))
        for key, m, p, color in zip(keys, measuredValues, predictedValues, colors):
            break
            rect_size = [ math.sqrt(k) * (math.log(plot_range[1]) - math.log(plot_range[0]))
                for k in key[0:3]
            ]
            rect_size[0] = 0.004 * rect_size[0] * m
            rect_size[1] = 0.004 * rect_size[1] * p
            rect_size.append( 0.004*rect_size[2] * p)
            rect_size[2] = 0.004 * rect_size[2] * m


            rect = patches.Rectangle(
                (m - rect_size[0], p - rect_size[1]),

                rect_size[0]*2.0,
                rect_size[1]*2.0,
                linewidth=1,
                edgecolor="black",
                facecolor="None",
                alpha =0.2,
            )
            ax.add_patch(rect)
            ax.plot( (m-rect_size[0],m+rect_size[0]), (p,p) , color=color, linewidth=1, alpha=0.3  )
            ax.plot( (m,m), (p - rect_size[1],p+rect_size[1]) , color=color , linewidth=1.0, alpha=0.3 )
            ax.plot( (m-rect_size[2],m+rect_size[2]), (p-rect_size[3],p+rect_size[3])  , color=color, linewidth=1.0, alpha=0.3 )

        #ax.scatter(
        #    measuredValues,
        #    predictedValues,
        #    s=[50] * len(colors),
        #    c=colors,
        #    marker=markers[r + 1],
        #    alpha=1,
        #    edgecolors="none",
        #    label=str(r),
        #)
    ax.set_xscale("log", base=1.5)
    ax.set_yscale("log", base=1.5)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot([0.1, 1150], [0.1, 1150], color="black", alpha=0.2)

    ax.set_xticks(
        [0.1, 1.0, 2, 4, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    )
    ax.set_yticks(
        [0.1, 1.0, 2, 4, 8, 9, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
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
            (max(min(lim[0] * 0.9, lim[0] - 0.1), 0.2), max(1, lim[1] * 1.1, lim[1] + 0.1))
        )
        ax.set_ylim(
            (max(min(lim[0] * 0.9, lim[0] - 0.1), 0.2), max(1, lim[1] * 1.1, lim[1] + 0.1))
        )

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel("actual Volume, B/Lup")
    ax.set_ylabel("predicted Volume, B/Lup")
    ax.legend()
    fig.tight_layout()
    ax.grid()

    plt.savefig( "./autoplots/" + title + ".svg")
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
    plt.savefig( "./autoplots/" + title + ".svg")
    plt.show()
