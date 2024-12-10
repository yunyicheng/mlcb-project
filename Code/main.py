import os
import sys

# import pisces.mapGRN as mapGRN
import pisces.mapProteinLigand as mapPL
import pisces.preprocessing as prep
from pisces.config import *
from pisces.train import *


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python3 main.py <value>")
    #     sys.exit(1)

    # # Access the argument
    # value = int(sys.argv[1])  # Convert to integer
    # mapPL.mapProteinLigand(value)
    # Example usage
    train_data_summary = prep.summarize_train_data()
    for key, value in train_data_summary.items():
        print(f"{key}: {value}")

    model = train_model()
    torch.save(model, OUTDIR / "final_model.pt")


if __name__ == "__main__":
    main()
