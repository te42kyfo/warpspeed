#!/usr/bin/env python3


def columnPrint(obj, columns):

    string = ""
    columnLabelWidths = [max([len(e) for e in c]) for c in columns]
    columnValueWidths = [max([ len((str(  int(obj.__dict__[e]) if isinstance( obj.__dict__[e], float) else obj.__dict__[e] ))) if e in obj.__dict__ else 5 for e in c]) for c in columns]




    rowCount = max([len(c) for c in columns])
    for row in range(rowCount):
        rowx = rowCount - row
        for col in range(len(columns)):
            if rowx <= len(columns[col]):
                string += obj.stringKey(columns[col][len(columns[col])-rowx], columnLabelWidths[col], columnValueWidths[col])
            else:
                string += obj.stringKey("", columnLabelWidths[col], columnValueWidths[col])
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
