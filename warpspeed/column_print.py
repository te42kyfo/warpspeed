#!/usr/bin/env python3


def generateColumns(obj, columns, referenceCount="perLup"):
    tableColumns = len(columns)
    tableRows = max([len(c) for c in columns])
    columnEntries = []
    columnAlignments = []

    for c in columns:
        columnEntries.append([])
        columnEntries.append([])
        columnEntries.append([])
        columnEntries[-3].extend([""] * (tableRows - len(c)))
        columnEntries[-2].extend([""] * (tableRows - len(c)))
        columnEntries[-1].extend([""] * (tableRows - len(c)))
        for e in c:
            columnEntries[-3].append(e[0].split(".")[-1] + ":")

            try:
                value = eval("obj." + e[0])
            except:
                value = "-"

            if isinstance(value, str):
                columnEntries[-2].append(value)
            elif isinstance(value, (list, tuple)):
                columnEntries[-2].append(str(value))
            else:
                if e[1] == "kB":
                    value /= 1024
                if e[1] == "MB":
                    value /= 1024 * 1024
                if e[1] == "GFlop/s":
                    flops = getattr(getattr(obj, "lc", obj), "flops", 0)
                    flops = getattr(obj, "flopsPerLup", 0)
                    flops = getattr(getattr(obj, "p", obj), "flopsPerLup", flops)
                    flops = getattr(getattr(obj, "lc", obj), "flops", flops)
                    if flops > 0:
                        value *= flops
                    else:
                        e = (e[0], "GLup/s")
                if e[1] == "B":
                    flops = getattr(obj, "flopsPerLup", 0)
                    flops = getattr(getattr(obj, "p", obj), "flopsPerLup", flops)
                    flops = getattr(getattr(obj, "lc", obj), "flops", flops)

                    if flops > 0:
                        value /= flops / 1000
                        e = (e[0], "mB/Flop")
                    else:
                        e = (e[0], "B/Lup")

                prec = (
                    0
                    if isinstance(value, int)
                    else 0 if value > 50 else 1 if value > 4 else 2
                )
                columnEntries[-2].append(
                    "{value:.{prec}f}".format(value=value, prec=prec)
                )
            columnEntries[-1].append(e[1])
        columnAlignments.extend([">", ">", "<"])

    return columnEntries, columnAlignments


def columnFormat(columnEntries, alignments, columnSeperator=" "):
    rows = len(columnEntries[0])

    columnWidths = []
    for c in columnEntries:
        width = 0
        for e in c:
            width = max(len(e), width)
        columnWidths.append(width)

    string = ""
    for row in range(len(columnEntries[0])):
        for c, width, alignment in zip(columnEntries, columnWidths, alignments):
            string += "{entry:{alignment}{width}}".format(
                entry=c[row], width=width, alignment=alignment
            )
            string += columnSeperator
        string += "\n"
    return string


def htlmColumnFormat(columnEntries, alignments, columnSeperator=" "):
    rows = len(columnEntries[0])

    string = '<table style="font-size:14px">\n'
    for row in range(len(columnEntries[0])):
        string += " <tr>\n"
        columnCounter = 0
        for c, alignment in zip(columnEntries, alignments):
            padding = 0.9 if columnCounter == 2 else 0.0
            string += (
                '  <td style="padding:0.4em;padding-right:'
                + str(padding)
                + "em;text-align:"
                + ("left" if alignment == "<" else "right")
                + '">'
                + ("<b>" if columnCounter == 0 else "")
                + c[row]
                + ("</b>" if columnCounter == 0 else "")
                + "</td>"
            )
            columnCounter = (columnCounter + 1) % 3
        string += " </tr>\n"
    string += "</table>"
    return string


def columnPrint(obj, columns):
    return columnFormat(*generateColumns(obj, columns))


def htmlColumnPrint(obj, columns, referenceCount="perLup"):
    return htlmColumnFormat(*generateColumns(obj, columns, referenceCount))
