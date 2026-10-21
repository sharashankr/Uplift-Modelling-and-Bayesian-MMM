# %% [markdown]
# Athey, Imbens, Metzger & Munro (2020) — reproducing Table 5
# ATT estimators on the real Lalonde-Dehejia-Wahba data.
# Units: thousands of 1978 dollars, as in the paper.

# %% Cell 1: load the authors' data
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data" if "__file__" in globals() else Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/gsbDBI/ds-wgan/master/data/original_data/{}.feather"
DESIGNS = {"E": "experimental", "CPS": "cps", "PSID": "psid"}

# The paper's eight pre-treatment variables (Section 3.2). u74/u75 are in the file but NOT used.
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]


def load_ldw(design):
    """Download once, cache locally; return t, the 8 covariates and re78, earnings in $1,000s."""
    path = DATA_DIR / f"ldw_{DESIGNS[design]}.feather"
    if not path.exists():
        r = requests.get(URL.format(DESIGNS[design]), timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_feather(path)
    df[["re74", "re75", "re78"]] /= 1000
    df["t"] = df["t"].astype(int)
    return df[["t", *COVARIATES, "re78"]]


data = {d: load_ldw(d) for d in DESIGNS}
for d, df in data.items():
    print(f"{d:5s} treated={(df.t == 1).sum():3d}  controls={(df.t == 0).sum():6,d}")