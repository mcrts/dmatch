#!/usr/bin/env python
# -*- coding: utf-8 -*
from sqlalchemy import create_engine
import pandas as pd

class MimicConnector:
    def __init__(self, uri):
        self.uri = uri

    def get_engine(self):
        return create_engine(self.uri)

    def get_terminology(self):
        query = """
            SELECT
                i.itemid as "entityid",
                i.label,
                i.fluid,
                i.category,
                i.loinc_code as loinc
            FROM d_labitems as i
            JOIN labevents e on e.itemid = i.itemid
            GROUP by (i.itemid, i.label, i.fluid, i.category, i.loinc_code)
            HAVING count(valuenum) > 1000
        """
        with self.get_engine().connect() as connection:
            df = pd.read_sql(query, connection)
        return df

    def get_terminology_aggregate(self):
        query = """
            SELECT
                i.itemid as entityid,
                count(1) as "size",
                avg(valuenum) as "mean",
                stddev(valuenum) as "std",
                variance(valuenum) as "var"
            FROM d_labitems as i
            JOIN labevents e on e.itemid = i.itemid
            GROUP by i.itemid
            HAVING count(valuenum) > 1000
        """
        with self.get_engine().connect() as connection:
            df = pd.read_sql(query, connection)
        df['frequency'] = 100 * df['size'] / sum(df['size'])
        df = df.sort_values('frequency', ascending=False)
        return df

    def get_entities(self, entityid):
        query = """
            SELECT
                valuenum as "value"
            FROM labevents
            WHERE
                itemid = {}
                and valuenum is not null
        """
        with self.get_engine().connect() as connection:
            df = pd.read_sql(query.format(entityid), connection)
        return df.value.to_numpy()


CONNECTORS = {
    'MIMIC3': MimicConnector,
}
