import pandas as pd
import sqlite3

def load_sensordata():
    conn = sqlite3.connect("../data/buildingData.db")
    return pd.read_sql_query(
        "select * from sensor_data where time is not null",
        conn,
        parse_dates="time",
        index_col=["time"],
    )

def load_sensordata_full():
    conn = sqlite3.connect("../data/buildingData2.db")
    return pd.read_sql_query(
        "select * from sensor_data where time is not null",
        conn,
        parse_dates="time",
        index_col=["time"],
    )


def filter_room(df, room):
    return df.loc[df.room == room].drop(["room"], axis=1)
