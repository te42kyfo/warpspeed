#!/usr/bin/env python3


def stringKey(self, key, labelWidth, valueWidth, siunit):

    labelWidth = max(len(key), labelWidth)

    value = getattr(self, key, "nav")

    if value == "nav":
        return "{:{labelWidth}}: {:{valueWidth}} {:7}".format(
            str(key),
            value,
            siunit,
            labelWidth=max(1,labelWidth),
            valueWidth=max(1, valueWidth))


    if isinstance(value, list):
        return "{:{labelWidth}}: {:{valueWidth}} ".format(
            str(key),
            str(value),
            labelWidth=labelWidth,
            valueWidth=max(1,valueWidth))

    if siunit == "kB":
        value /= 1024

    if siunit == "MB":
        value /= 1024*1024

    prec = 0 if isinstance(value, str) or isinstance(value, int) else 1 if value < 50 else 0

    return "{:{labelWidth}}: {:{valueWidth}.{prec}f} {:7}".format(
            str(key),
            value,
            siunit,
            labelWidth=labelWidth,
            valueWidth=valueWidth-12,
            prec=prec)

#def columnPrint(obj, columns):

#    string = ""
#    columnLabelWidths = [max([len(e[0]) for e in c]) for c in columns]
#    columnValueWidths = [max([len(stringKey(obj, e[0], 0, 0, e[1])) - len(e[0]) for e in c]) for c in columns]

#    rowCount = max([len(c) for c in columns])
#    for row in range(rowCount):
#        rowx = rowCount - row
#        for col in range(len(columns)):
#            if rowx <= len(columns[col]):
#                string += stringKey(obj, columns[col][len(columns[col])-rowx][0], columnLabelWidths[col], columnValueWidths[col], columns[col][len(columns[col])-rowx][1])
#            else:
#                string += stringKey(obj, "", columnLabelWidths[col], columnValueWidths[col], "")
#        string += "\n"
#    return string


def columnPrint(obj, columns):

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

            columnEntries[-3].append( e[0].split(".")[-1] + ":")
            value = eval("obj." + e[0])

            if isinstance(value, str):
                columnEntries[-2].append( value )
            elif isinstance(value, (list, tuple)):
                columnEntries[-2].append( str(value) )
            else:
                if e[1] == "kB":
                    value /= 1024
                if e[1] == "MB":
                    value /= 1024*1024
                prec = 0 if isinstance(value, int) else 1 if value < 50 else 0
                columnEntries[-2].append( "{value:.{prec}f}".format(value=value, prec=prec))

            columnEntries[-1].append( e[1] + " " )
        columnAlignments.extend([">", ">", "<"])

    
    return columnFormat(columnEntries, columnAlignments, " ")


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
            string += "{entry:{alignment}{width}}".format(entry = c[row], width=width, alignment=alignment)
            string += columnSeperator
        string += "\n"
    return string


    


def htmlColumnPrint(obj, columns):

    string = "<table style=\"padding:10px\">"
    columnLabelWidths = [max([len(e) for e in c]) for c in columns]
    columnValueWidths = [max([ len((str(  int(obj.__dict__[e]) if isinstance( obj.__dict__[e], float) else obj.__dict__[e] ))) if e in obj.__dict__ else 5 for e in c]) for c in columns]




    rowCount = max([len(c) for c in columns])
    for row in range(rowCount):
        string += "<tr>"
        rowx = rowCount - row
        for col in range(len(columns)):
            string += "<td style=\"padding:10px;text-align:right;\">"
            if rowx <= len(columns[col]):
                string += columns[col][len(columns[col]) -rowx]
                string += "</td><td style=\"padding:10px\">"
                string += str( obj.__dict__[columns[col][len(columns[col]) -rowx]] )
            else:
                string += "</td><td>"
            string += "</td>"
        string += "</tr>"


    string += "</table>"
    return string



def formattedHtmlColumnPrint(obj, columns):

    string = "<table style=\"padding:10px\">"

    rowCount = max([len(c) for c in columns])
    for row in range(rowCount):
        string += "<tr>"
        rowx = rowCount - row
        for col in range(len(columns)):
            string += "<td style=\"padding:10px; text-align:right\">"
            cell = columns[col][len(columns[col]) -rowx]
            if rowx <= len(columns[col]):
                string += cell[0]
                string += ": </td><td style=\"padding:10px; background-color:#DDDDDD\">"
                string += cell[1](obj.__dict__[cell[0]])
            else:
                string += "</td><td>"
            string += "</td>"
        string += "</tr>"


    string += "</table>"
    return string
