from pathlib import Path

HOMEDIR = Path(__file__).resolve().parents[2]
DATADIR = HOMEDIR / 'Data'
OUTDIR = HOMEDIR / 'Results'
GRAPHOUTDIR = OUTDIR / 'Graph'
INTERDATADIR = DATADIR / 'Intermediate'
