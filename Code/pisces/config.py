from pathlib import Path

HOMEDIR = Path(__file__).resolve().parents[2]
DATADIR = HOMEDIR / 'Data'
OUTDIR = HOMEDIR / 'Results'
INTERDATADIR = DATADIR / 'Intermediate'
