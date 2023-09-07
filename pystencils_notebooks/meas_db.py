#!/usr/bin/env python3
import sys

sys.path.append("../pystencils")
sys.path.append("../genpredict")

import json
import sqlite3
from predict_metrics import *
from measured_metrics import MeasuredMetrics, ResultComparer


class MeasDB:
    def __init__(self, dbName):
        self.conn = sqlite3.connect(dbName)
        self.conn.row_factory = sqlite3.Row
        self.c = self.conn.cursor()

        query = "SELECT COUNT(*) FROM measurements"
        c = self.conn.cursor()
        c.execute(query)
        print(str(c.fetchone()[0]) + " entries in db")

    def clearDB(self):
        query = "SELECT COUNT(*) FROM measurements"
        c = self.conn.cursor()
        c.execute(query)
        print("clear DB")
        print(str(c.fetchone()[0]) + " entries before")

        query = "DELETE FROM measurements"
        self.conn.execute(query)
        self.conn.isolation_level = None
        self.conn.execute("VACUUM")
        self.conn.isolation_level = ""

        query = "SELECT COUNT(*) FROM measurements"
        c = self.conn.cursor()
        c.execute(query)
        print(str(c.fetchone()[0]) + " entries after")
        self.conn.commit()

    def insertValue(
        self, stencilRange, block, threadFolding, device, basic, meas, lc, fieldSize
    ):
        self.insertValueKeys(
            {
                "range": stencilRange,
                "blockx": block[0],
                "blocky": block[1],
                "blockz": block[2],
                "tfoldx": threadFolding[0],
                "tfoldy": threadFolding[1],
                "tfoldz": threadFolding[2],
                "device": '"' + str(device.name) + '"',
            },
            {
                "domainx": fieldSize[0],
                "domainy": fieldSize[1],
                "basic_metrics": "'" + json.dumps(basic.__dict__) + "'",
                "measured_metrics": "'" + json.dumps(meas.__dict__) + "'",
                "launch_config": "'" + json.dumps(lc.__dict__) + "'",
            },
        )

    def insertValueKeys(self, keys, data):
        query = "DELETE FROM measurements WHERE "
        query += " AND ".join(["{}={}".format(key, keys[key]) for key in keys])
        self.conn.execute(query)

        query = "INSERT INTO measurements"
        query += "(" + ", ".join([k for k, v in (keys | data).items()]) + ") VALUES "
        query += "(" + ", ".join([str(v) for k, v in (keys | data).items()]) + ")"

        self.conn.execute(query)

    def getBasicMetricsRange(self, stencilRange, device):
        query = (
            "SELECT * FROM measurements WHERE " 'range={} AND device="{}" '
        ).format(stencilRange, device.name)
        c = self.conn.cursor()
        c.execute(query)

        return [
            (
                LaunchConfig.fromDict(json.loads(row["launch_config"])),
                BasicMetrics.fromDict(json.loads(row["basic_metrics"])),
                MeasuredMetrics.fromDict(json.loads(row["measured_metrics"])),
            )
            for row in c.fetchall()
        ]

    def getEntry(self, stencilRange, block, threadFolding, fieldSize, device):
        return self.getEntryKeys(
            {
                "range": stencilRange,
                "blockx": block[0],
                "blocky": block[1],
                "blockz": block[2],
                "tfoldx": threadFolding[0],
                "tfoldy": threadFolding[1],
                "tfoldz": threadFolding[2],
                "device": '"' + str(device.name) + '"',
            }
        )

    def getEntryKeys(self, keys):
        query = "SELECT * FROM measurements WHERE "
        query += " AND ".join(["{}={}".format(key, keys[key]) for key in keys])

        c = self.conn.cursor()

        c.execute(query)
        row = c.fetchone()
        if row is None:
            return None, None, None

        secondResult = c.fetchone()
        if secondResult is not None:
            print("Duplicate Result!")
            return None, None, None

        return (
            LaunchConfig.fromDict(json.loads(row["launch_config"])),
            BasicMetrics.fromDict(json.loads(row["basic_metrics"])),
            MeasuredMetrics.fromDict(json.loads(row["measured_metrics"])),
        )

    def commit(self):
        self.conn.commit()
