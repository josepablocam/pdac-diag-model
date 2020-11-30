# Use https://github.com/jackwasey/icd
# to extract some amount of readable ICD9/10 description
# so that Limor can easily valid things manually
# (rather than have to repeatedly search things online)

import pandas as pd

from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

icdr = importr("icd")
rinterpreter = robjects.r

pandas2ri.activate()

rinterpreter.source("R/icd_helpers.R")
get_possible_codes_ = rinterpreter("get_possible_codes")
get_codes_by_pattern_ = rinterpreter("get_codes_by_pattern")


def get_possible_codes(include=None, exclude=None):
    if include is None:
        include = []
    if exclude is None:
        exclude = []
    results = get_possible_codes_(include, exclude)
    # single explanation for each code
    return results.groupby("code").head(1).reset_index(drop=True)


def get_codes_by_pattern(pattern):
    results = get_codes_by_pattern_(pattern)
    return results.groupby("code").head(1).reset_index(drop=True)


def get_code_explanation(code):
    # using icdr.explain_code seems to produce some odd
    # results when it gets confused about whether a code is
    # icd9 or icd10
    # icdr.explain_code("E13.9")
    #     array(['Other household maintenance'], dtype='<U27')
    #
    # icdr.explain_code_icd10("E13.9")
    # array(['Other specified diabetes mellitus without complications'],
    #       dtype='<U55')
    #
    # icdr.explain_code_icd9("E13.9")
    #     array(['Other household maintenance'], dtype='<U27')
    # whereas going through get_codes_by_pattern seems to avoid this problem
    # altogether
    if "." in code:
        # remove period, messes with regex below
        code = code.replace(".", "")
    results = get_codes_by_pattern("^{}$".format(code))
    explanation = results.long_desc
    if len(explanation) == 0:
        return None
    else:
        return str(explanation[0])


# Taken from https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion
ICD_MAPPING_DF = None
def build_mapping_table():
    global ICD_MAPPING_DF
    if ICD_MAPPING_DF is not None:
        return
    ICD_MAPPING_FILES = "resources/ICD10-ICD9-codes-conversion/ICD_9_10_d_v1.1.csv"
    ICD_MAPPING_DF = pd.read_csv(
        ICD_MAPPING_FILES,
        sep="|",
        header=0,
        names=["icd10", "icd9", "other"],
    )


def robust_decimal_to_short_icd9(code):
    short = str(icdr.decimal_to_short_icd9(code)[0])
    if len(short) > 0:
        return short
    else:
        return code.replace(".", "")


def convert_code_(code, src_type):
    build_mapping_table()
    if src_type == "icd9":
        src_expand = icdr.short_to_decimal_icd9
        target = "icd10"
        target_shorten = icdr.decimal_to_short_icd10
    else:
        src_expand = icdr.short_to_decimal_icd10
        target = "icd9"
        target_shorten = icdr.decimal_to_short_icd9

    if "." not in code:
        # looking up requires decimal version
        code = str(src_expand(code)[0])

    mapped = ICD_MAPPING_DF[ICD_MAPPING_DF[src_type] == code][target].values
    clean = []
    for c in mapped:
        # return short version
        c = str(target_shorten(c)[0])
        if len(c) > 0:
            # best effort...
            c = c.replace(".", "")
        clean.append(c)
    return sorted(clean)


def icd9_to_icd10(code):
    return convert_code_(code, "icd9")


def icd10_to_icd9(code):
    return convert_code_(code, "icd10")
