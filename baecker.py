# We implement proxy to the Baecker model, see reference below.
# We use the ICD9 codes (and corresponding ICD10 counterparts)
# based on the paper below
# @article{baecker2019changes,
#   title={Do changes in health reveal the possibility of undiagnosed pancreatic cancer? Development of a risk-prediction model based on healthcare claims data},
#   author={Baecker, Aileen and Kim, Sungjin and Risch, Harvey A and Nuckols, Teryl K and Wu, Bechien U and Hendifar, Andrew E and Pandol, Stephen J and Pisegna, Joseph R and Jeon, Christie Y},
#   journal={PloS one},
#   volume={14},
#   number={6},
#   pages={e0218580},
#   year={2019},
#   publisher={Public Library of Science}
# }
# We also add race indicator, gender indicator and
# 5 year age buckets
# replicating the model described in
# Note that we * do not * have a way of adding
# influenza vaccine status easily, so we elide this
# and note in the limitations section of paper

from argparse import ArgumentParser
from collections import OrderedDict
import pickle
import re

import pandas as pd

import icd_r_wrapper as icd


def get_codes(include_pattern, exclude_pattern=None):
    df = icd.get_codes_by_pattern(include_pattern)
    if exclude_pattern is None:
        return df.code.values.tolist()
    exclude = df.code.map(
        lambda x: re.match(exclude_pattern, x) is not None
    ).values
    return df[~exclude].code.values.tolist()


def build_baecker_feature_map():
    """
    Returns a map from baecker feature to set of ICD-9/10 codes
    for that feature
    """
    icd9_codes = {
        "acute-pancreatitis": ["577.0"],
        "chronic-pancreatitis": ["577.1"],
        "diabetes":
        get_codes("^250", "^250.[13]"),
        "poorly-controlled-dm":
        get_codes("^250", "(^250.[13])|(^25000$)"),
        "dyspesia-gastritis-peptic-ucler":
        ["536.8", "535.5", "533.90", "533.91"] +
        get_codes("(^533[0-5])|(5336[01])|(5336)"),
        "gallbladder-disease": ["575.9"],
        "acute-cholecystitis": ["575.0"],
        "depression": [
            "311", "2962", "2963", "2965", "2966", "2967", "2980", "30110",
            "30113", "3090", "3091"
        ],
        "abdominal-pain":
        get_codes("^7890"),
        "upper-abdominal-pain": ["789.01", "789.02", "789.06"],
        "chest-pain": ["413.9"] + get_codes("^7865"),
        "abnormal-feces": ["787.7"],
        "other-digestive-symptoms": ["787.99"],
        "flatulence": ["787.3"],
        "change-bowel-habits": ["787.9"],
        "constipation": ["564.0"],
        "diarrhea": ["787.91"],
        "irritable-bowel": ["536.9"],
        "esophageal-reflux": ["530.81"],
        "jaundice": ["782.4"],
        "anorexia": ["783.0"],
        "abnormal-weight-loss": ["783.21", "783.22"],
        "cachexia": ["799.4"],
        "feeding-difficulties": ["783.3"],
        "nausea-vomiting": ["787.01"],
        "nausea-alone": ["787.02"],
        "vomiting-alone": ["787.03"],
        "malaise": ["780.7"],
        "itching": ["698.9"],
    }
    icd9_codes = {
        label: set([icd.robust_decimal_to_short_icd9(c) for c in codes])
        for label, codes in icd9_codes.items()
    }
    icd10_codes = {
        label:
        set([mapped_c for c in codes for mapped_c in icd.icd9_to_icd10(c)])
        for label, codes in icd9_codes.items()
    }
    feat_map = {
        label: icd9_codes[label].union(icd10_codes[label])
        for label in icd9_codes.keys()
    }
    # make sure dictionary stays in the order provided
    return OrderedDict(feat_map)


def get_args():
    parser = ArgumentParser(description="Generate feature map used in baecker")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled feature map",
    )
    return parser.parse_args()


def main():
    args = get_args()
    feat_map = build_baecker_feature_map()
    with open(args.output, "wb") as fout:
        pickle.dump(feat_map, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
