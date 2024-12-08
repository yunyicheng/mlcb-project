import os

from pisces.config import *
import pisces.preprocessing as prep
import pisces.mapGRN as mapGRN


def main():
    prep.load_data()
    
    os.makedirs(INTERDATADIR, exist_ok=True)

    mapGRN.map_grn()


if __name__ == "__main__":
    main()
