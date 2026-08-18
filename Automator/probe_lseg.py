"""
probe_lseg.py — minimal LSEG connectivity/session smoke test.
LSEG equivalent of ICEBREAKER/COT_ALL/Automator/probe_kc.py (which tests
icepython connectivity). Run manually to confirm the LSEG Workspace API
session is reachable before a scheduled run.
"""

import lseg.data as ld

if __name__ == "__main__":
    ld.open_session()
    try:
        df = ld.get_history(universe=["4083731CLNG"], fields=["COMM_LAST"],
                             start="2024-01-01", end="2024-03-31", interval="daily")
        if df is not None and not df.empty:
            print("OK — LSEG session reachable, sample rows:")
            print(df.tail(3))
        else:
            print("WARNING — session opened but query returned no data.")
    finally:
        ld.close_session()
