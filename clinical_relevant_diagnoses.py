import icd_r_wrapper


def code_str_to_list(codes_str):
    code_values = [c.strip() for c in codes_str.split()]
    # remove empty values
    code_values = [c for c in code_values if len(c) > 0]
    # note that the BIDMC data in Diagnoses.txt actually removes the period
    code_values = [c.replace(".", "") for c in code_values]
    return code_values


CODES = {}

pancreatic_duct_adenocarcinoma = """
157
157.0
157.1
157.2
157.3
157.8
157.9

C25
C25.0
C25.1
C25.2
C25.3
C25.7
C25.8
C25.9
"""
CODES["pancreatic_duct_adenocarcinoma"] = pancreatic_duct_adenocarcinoma

diabetes_mellitus = """
250
250.0
250.00
250.01
250.02
250.03
E11
E11.9
E13.65
"""
CODES["diabetes_mellitus"] = diabetes_mellitus

dm_with_complications = """
250.1
250.10
250.11
250.12
250.13
250.2
250.20
250.21
250.22
250.23
250.3
250.30
250.31
250.32
250.33
250.4
250.40
250.41
250.42
250.43
250.5
250.50
250.51
250.52
250.53
250.6
250.60
250.61
250.62
250.63
250.7
250.70
250.71
250.72
250.73
250.8
250.80
250.81
250.82
250.83
250.9
250.90
250.91
250.92
250.93

E11.0
E11.00
E11.01
E11.2
E11.21
E11.22
E11.29
E11.3
E11.36
E11.4
E11.40
E11.41
E11.42
E11.43
E11.44
E11.49
E11.5
E11.51
E11.52
E11.59
E11.6
E11.61
E11.610
E11.618
E11.65
E11.64
E11.641
E11.649
E11.63
E11.630
E11.69
E11.62
E11.620
E11.621
E11.628
E11.622
E11.8
"""
CODES["diabetes_mellitus_with_complications"] = dm_with_complications

prediabetic = """
790.21
790.22
R73.01
"""
CODES["prediabetic"] = prediabetic

hyperglycemia = """
790.2
790.29
R73
R73.0
R73.02
R73.9
R73.09
"""
CODES["hyperglycemia"] = hyperglycemia

abdominal_pain = """
789.0
789.00
789.01
789.02
789.03
789.04
789.05
789.06
789.07
789.09

R10
R10.8
R10.84
R10.3
R10.30
R10.31
R10.32
R10.33
R10.1
R10.9
R10.10
R10.11
R10.12
R10.13
"""
CODES["abdominal_pain"] = abdominal_pain

essential_hypertension = """
405.99
401
401.1
401.9
I10
"""
CODES["essential_hypertension"] = essential_hypertension

esophageal_reflux = """
530.81
530.11
K21.9
K21
K21.0
"""
CODES["esophageal_reflux"] = esophageal_reflux

hyperlipidemia = """
272.4
272.2
E78.2
E78.5
E78.4
"""
CODES["hyperlipidemia"] = hyperlipidemia

hypercholesterolemia = """
272.0
E78.0
E78.00
"""
CODES["hypercholesterolemia"] = hypercholesterolemia

routine = """
V70.0
Z00.00
"""
CODES["routine_visit"] = routine

anemia = """
285.9
280.9
280.0
281.9
D64.9
D509
"""
CODES["anemia"] = anemia

abnormal_weight_loss = """
783.2
783.21
R63.4
"""
CODES["abnormal_weight_loss"] = abnormal_weight_loss

hist_tobacco_use = """
V15.82
Z87.891
"""
CODES["hist_tobacco_use"] = hist_tobacco_use

tobacco_use_disorder = """
305.1
F17.20
F17.21
"""
CODES["tobacco_use_disorder"] = tobacco_use_disorder

obstruction_bile_duct = """
576.2
K83.1
"""
CODES["obstruction_bile_duct"] = obstruction_bile_duct

other_preop = """
V72.83
Z01818
"""
CODES["other_preop"] = other_preop

preop_cv = """
V72.81
Z01810
"""
CODES["preop_cv"] = preop_cv

pancreas_unspec = """
577.9
K869
"""
CODES["pancreas_unspec"] = pancreas_unspec

hyperthyroidism = """
244.0
244.9
E03.9
"""
CODES["hyperthyroidism"] = hyperthyroidism

liver_metastases = """
197.7
C78.7
"""
CODES["liver_metastases"] = liver_metastases

atherosclerosic_heart_disease = """
414.01
I25.10
"""
CODES["atherosclerosic_heart_disease"] = atherosclerosic_heart_disease

cyst_pancreas = """
577.2
K86.2
K86.3
"""
CODES["cyst_pancreas"] = cyst_pancreas

long_term_anticoagulation = """
V58.61
Z79.01
"""
CODES["long_term_anticoagulation"] = long_term_anticoagulation

chest_pain = """
786.50
R07.9
"""
CODES["chest_pain"] = chest_pain

atrial_fibrillation = """
427.31
I48.9
"""
CODES["atrial_fibrillation"] = atrial_fibrillation

malaise_and_fatigue = """
780.7
780.79
R53.8
R53.81
R53.82
R53.83
"""
CODES["malaise_and_fatigue"] = malaise_and_fatigue

chronic_pancreatitis = """
57.71
K861
"""
CODES["chronic_pancreatitis"] = chronic_pancreatitis

jaundice = """
7824
R17
"""
CODES["jaundice"] = jaundice

chronic_ischemic_heart = """
41400
4148
414
4149
"""
CODES["chronic_ischemic_heart"] = chronic_ischemic_heart

diarrhea = """
78791
R197
"""
CODES["diarrhea"] = diarrhea

cholangitis = """
5761
K830
"""
CODES["cholangitis"] = cholangitis

anxiety = """
30000
F419
30928
30924
"""
CODES["anxiety"] = anxiety

nausea = """
78701
78702
R110
R112
"""
CODES["nausea"] = nausea

calculus_gallbladder = """
57420
57410
57490
K8020
"""
CODES["calculus_gallbladder"] = calculus_gallbladder

# based on Wazir et al (Yale) paper

emphysema = """
492.0
492.8
J43.9
J43.2
"""
CODES["emphysema"] = emphysema

asthma = """
J45.90
J45.99
J45.20
J45.41
J45.40
J45.30
J45.31
J45.50
J45.21
493.90
493.92
493.00
493.10
493.12
493.02
493.91
"""
CODES["asthma"] = asthma

stroke = """
I639
I6349
I6313
I6341
I6343
I6342
I6340
I6323
I6310
I6302
I636
I638
I6351
I6359
I6344
I6311
I6353
I6333
I6330
I6350
I6352
I6303
I6331
430
431
4320
4329
4321
43330
43321
43311
43310
43331
43320
43300
43390
43380
43301
43491
43411
43410
43490
43400
43401
4359
4353
4352
4358
4351
"""
CODES["stroke"] = stroke

# based on
# https://www.questdiagnostics.com/dms/Documents/Other/PDF_MI4632_ICD_9-10_Codes_for_Cardio_38365_062915.pdf
# in addition to atherosclerosic_heart_disease
coronary = """
414.4
I25.84
"""
CODES["coronary_heart_disease"] = coronary

angina = """
413
413.0
413.1
413.9
I20
I20.0
I20.1
I20.8
I20.9
"""
CODES["angina_pectoris"] = angina

heart_attack = """
41081
41071
41092
41041
41031
41001
41091
41080
4109
41072
41011
41090
41022
4100
41000
41012
41002
41042
41051
I214
I2109
I2111
I2119
I21A1
I213
I2102
I222
I221
I229
I237
I238
I232
"""
CODES["heart_attack"] = heart_attack

other_heart_disease = """
I519
I517
I513
I5189
I5181
"""
CODES["other_heart_disease"] = other_heart_disease

ulcer = """
53190
53170
53140
53191
53100
53130
53171
53120
53150
53141
53290
53240
53270
53231
53291
53260
53261
53250
53241
53200
53370
53390
53300
53340
53440
53490
53491
53410
K259
K254
K250
K255
K253
K251
K269
K264
K261
K265
K279
K277
K274
"""
CODES["ulcer"] = ulcer

# technically these are family history of malignant
# neoplasm of digestive organs/gastro tract
family_history = """
V160
Z800
"""

CODES["family_history_pancreas_cancer"] = family_history

alcohol_abuse = """
3050
30500
30501
30502
30503
"""
alcohol_abuse += "\n" + ("\n".join(
    icd_r_wrapper.get_codes_by_pattern("^F10").code.values))
CODES["alcohol_abuse"] = alcohol_abuse

exercise = "\n".join(
    icd_r_wrapper.get_codes_by_pattern("^Y93([0-7]|[AB])").code.values)
CODES["exercise"] = exercise


# other cancer history
def check_cancer_code(code):
    # Limors suggested ranges are
    # 140-239.99
    # c00-d49
    if code[0].isdigit():
        return 140 <= int(code[:3]) <= 239
    else:
        if code[0] not in ["C", "D"]:
            return False
        else:
            if code[0] == "D":
                return int(code[1]) <= 4
            else:
                return True


def get_other_cancer_history():
    codes_df = icd_r_wrapper.get_possible_codes(
        include=["(neoplasm)|(cancer)|(tumor)"], )

    is_cancer = codes_df.code.map(check_cancer_code)
    codes_df = codes_df[is_cancer]
    # drop anything that we have in pancreatic cancer or liver liver_metastases
    exclude = []
    exclude += code_str_to_list(CODES["pancreatic_duct_adenocarcinoma"])
    exclude += code_str_to_list(CODES["liver_metastases"])
    codes_df = codes_df[~codes_df.code.isin(exclude)]
    return "\n".join(codes_df.code.str.upper().values)


other_cancer_history = get_other_cancer_history()
CODES["other_cancer_history"] = other_cancer_history

for label in CODES.keys():
    CODES[label] = code_str_to_list(CODES[label])

CANONICAL_CODES = {}
already_observed = set([])
for label, code_values in CODES.items():
    if not already_observed.isdisjoint(code_values):
        overlap = already_observed.intersection(code_values)
        msg = "ICD9/10 values must be disjoint across labels"
        msg += "\n Overlaps for: {}".format(overlap)
        raise Exception(msg)
    already_observed.update(code_values)
    CANONICAL_CODES.update({v: label for v in code_values})

_missing = set([v for ls in CODES.values()
                for v in ls]).difference(CANONICAL_CODES.keys())

assert len(_missing) == 0, "Some codes are not present in CANONICAL_CODES"


def get_codes():
    return CODES


def get_canonical_codes():
    return CANONICAL_CODES
