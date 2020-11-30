#!/usr/bin/env bash
source folder_setup.sh

mkdir -p ${ANALYSIS_DIR}
mkdir -p ${DATA_DIR}
mkdir -p ${RESULTS_DIR}

cp /mnt/LargeControlGroupDiagnoses.txt "${DATA_DIR}/BIDMC-Controls.txt"
cp /mnt/Diagnoses.txt "${DATA_DIR}/BIDMC-Cancer.txt"
cp /mnt/verified-patients.csv "${DATA_DIR}/BIDMC-verified-cancer.csv"

cp /mnt/LargeControlGroupDemographics.txt "${DATA_DIR}/BIDMC-Controls-Demo.txt"
cp /mnt/Demographics.txt "${DATA_DIR}/BIDMC-Cancer-Demo.txt"


cp /mnt/PHCcontrolsDiagnoses.txt "${DATA_DIR}/PHC-Controls.txt"
cp /mnt/PHCcohortDiagnoses.txt "${DATA_DIR}/PHC-Cancer.txt"

cp /mnt/PHCcontrolsDemographics.txt "${DATA_DIR}/PHC-Controls-Demo.txt"
cp /mnt/PHCcohortDemographics.txt "${DATA_DIR}/PHC-Cancer-Demo.txt"


#### Data Preparation #######
python preprocess_demographics.py \
    --which BIDMC \
    --input "${DATA_DIR}/BIDMC-Cancer-Demo.txt" "${DATA_DIR}/BIDMC-Controls-Demo.txt" \
    --output "${DATA_DIR}/BIDMC-demo.feather"

python preprocess.py \
  --which BIDMC \
  --controls "${DATA_DIR}/BIDMC-Controls.txt" \
  --cancer "${DATA_DIR}/BIDMC-Cancer.txt" \
  --verified "${DATA_DIR}/BIDMC-verified-cancer.csv" \
  --n_heuristic_filter 180 \
  --output "${DATA_DIR}/BIDMC.feather"

# only sample from those that have satisfied our heuristic filter
# to avoid dropping controls later
python patient_matching.py \
    --cancer "${DATA_DIR}/BIDMC-Cancer-Demo.txt" \
    --controls "${DATA_DIR}/BIDMC-Controls-Demo.txt" \
    --filtered_diagnosis_data "${DATA_DIR}/BIDMC.feather" \
    --seed 42 \
    --output "${DATA_DIR}/BIDMC-matched-UIDs.pkl"


python patient_filtering.py \
    --uid  "${DATA_DIR}/BIDMC-matched-UIDs.pkl" \
    --input \
      "${DATA_DIR}/BIDMC.feather" \
      "${DATA_DIR}/BIDMC-demo.feather" \
    --output \
      "${DATA_DIR}/BIDMC-matched.feather" \
      "${DATA_DIR}/BIDMC-demo-matched.feather"


python split_train.py \
  --input "${DATA_DIR}/BIDMC-matched.feather" \
  --test_size 0.2 \
  --seed 42 \
  --output "${DATA_DIR}"


python min_count_diagnoses.py \
    --input "${DATA_DIR}/BIDMC-matched-train.feather" \
    --min_count 100 \
    --output "${DATA_DIR}/BIDMC-train-diagnoses.pkl"


## PHC Data ##
python preprocess.py \
  --which PHC \
  --controls "${DATA_DIR}/PHC-Controls.txt" \
  --cancer "${DATA_DIR}/PHC-Cancer.txt" \
  --n_heuristic_filter 180 \
  --output "${DATA_DIR}/PHC.feather"


python preprocess_demographics.py \
    --which PHC \
    --input "${DATA_DIR}/PHC-Cancer-Demo.txt" "${DATA_DIR}/PHC-Controls-Demo.txt" \
    --output "${DATA_DIR}/PHC-demo.feather"


### Baecker preparation ###
python baecker.py \
    --output "${DATA_DIR}/baecker-diagnoses.pkl"


##### Sweep for model hyperparameters #####
mkdir -p ${RESULTS_DIR}/cv-search/
python cv_search.py \
  --input "${DATA_DIR}/BIDMC-matched-train.feather" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --models lr nn \
  --cutoffs 180 270 365 \
  --n_obs_sample 10000 \
  --n_iters 20 \
  --max_configs 20 \
  --seed 42 \
  --output_dir "${RESULTS_DIR}/cv-search/"


#### Train set performance #####
mkdir -p "${RESULTS_DIR}/bidmc-train-results/"

python validate.py \
  --mode train-performance \
  --models lr nn \
  --train "${DATA_DIR}/BIDMC-matched-train.feather" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --hyperparameters "${RESULTS_DIR}/cv-search/params-summary.pkl" \
  --n_splits 10 \
  --cutoffs 180 270 365 \
  --output "${RESULTS_DIR}/bidmc-train-results/ours.pkl" \
  --seed 42 \
  --bootstrap_iters 1000

python validate.py \
  --mode train-performance \
  --models baecker \
  --features "${DATA_DIR}/baecker-diagnoses.pkl" \
  --train "${DATA_DIR}/BIDMC-matched-train.feather" \
  --train_demographics "${DATA_DIR}/BIDMC-demo-matched.feather" \
  --n_splits 10 \
  --cutoffs 180 270 365 \
  --output "${RESULTS_DIR}/bidmc-train-results/baecker.pkl" \
  --seed 42 \
  --bootstrap_iters 1000


##### Train models with all of BIDMC train ####
mkdir -p "${RESULTS_DIR}/bidmc-trained-models/"

# we train them one cutoff at a time and save down
for cutoff in 180 270 365
do
  python validate.py \
    --mode just-train \
    --models lr nn \
    --train "${DATA_DIR}/BIDMC-matched-train.feather" \
    --hyperparameters "${RESULTS_DIR}/cv-search/params-summary.pkl" \
    --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
    --cutoffs ${cutoff} \
    --output "${RESULTS_DIR}/bidmc-trained-models/ours-${cutoff}.pkl" \
    --seed 42
done

# put models together in single list for ease of use
python combine_trained_models_list.py \
    --input ${RESULTS_DIR}/bidmc-trained-models/ours-*.pkl \
    --output "${RESULTS_DIR}/bidmc-trained-models/ours.pkl"


### Test set performance ####
mkdir -p "${RESULTS_DIR}/bidmc-test-results/"

python validate.py \
  --mode test-performance \
  --models lr nn \
  --trained_models "${RESULTS_DIR}/bidmc-trained-models/ours.pkl" \
  --test "${DATA_DIR}/BIDMC-matched-test.feather" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --cutoffs 180 270 365 \
  --output "${RESULTS_DIR}/bidmc-test-results/ours.pkl" \
  --seed 42 \
  --bootstrap_iters 1000

python validate.py \
  --mode test-performance \
  --models baecker \
  --train "${DATA_DIR}/BIDMC-matched-train.feather" \
  --test "${DATA_DIR}/BIDMC-matched-test.feather" \
  --features "${DATA_DIR}/baecker-diagnoses.pkl" \
  --train_demographics "${DATA_DIR}/BIDMC-demo-matched.feather" \
  --test_demographics "${DATA_DIR}/BIDMC-demo.feather" \
  --cutoffs 180 270 365 \
  --output "${RESULTS_DIR}/bidmc-test-results/baecker.pkl" \
  --seed 42 \
  --bootstrap_iters 1000

### Train on entire BIDMC for external evaluation ####
mkdir -p "${RESULTS_DIR}/bidmc-full-trained-models/"
# we train them one cutoff at a time and save down
for cutoff in 180 270 365
do
  python validate.py \
    --mode just-train \
    --models lr nn \
    --train "${DATA_DIR}/BIDMC-matched.feather" \
    --hyperparameters "${RESULTS_DIR}/cv-search/params-summary.pkl" \
    --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
    --cutoffs ${cutoff} \
    --output "${RESULTS_DIR}/bidmc-full-trained-models/ours-${cutoff}.pkl" \
    --seed 42
done

# put models together in single list for ease of use
python combine_trained_models_list.py \
    --input ${RESULTS_DIR}/bidmc-full-trained-models/ours-*.pkl \
    --output "${RESULTS_DIR}/bidmc-full-trained-models/ours.pkl"


### External population performance ####
mkdir -p "${RESULTS_DIR}/phc-test-results"

# we use the models trained on BIDMC full
# Note: we take all PHC rather than 10-fold CV
# to address reviewer feedback
python validate.py \
  --mode external-performance \
  --models lr nn \
  --trained_models "${RESULTS_DIR}/bidmc-full-trained-models/ours.pkl" \
  --test "${DATA_DIR}/PHC.feather" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --cutoffs 180 270 365 \
  --n_splits 0 \
  --output "${RESULTS_DIR}/phc-test-results/ours.pkl" \
  --seed 42 \
  --bootstrap_iters 1000

python validate.py \
  --mode external-performance \
  --models baecker \
  --train "${DATA_DIR}/BIDMC-matched.feather" \
  --test "${DATA_DIR}/PHC.feather" \
  --features "${DATA_DIR}/baecker-diagnoses.pkl" \
  --train_demographics "${DATA_DIR}/BIDMC-demo-matched.feather" \
  --test_demographics "${DATA_DIR}/PHC-demo.feather" \
  --cutoffs 180 270 365 \
  --n_splits 0 \
  --output "${RESULTS_DIR}/phc-test-results/baecker.pkl" \
  --seed 42 \
  --bootstrap_iters 1000


### External population performance when re-trained ####
mkdir -p "${RESULTS_DIR}/phc-retrain-results"
# re train models on PHC and test there
python validate.py \
  --mode train-performance \
  --models lr nn \
  --hyperparameters "${RESULTS_DIR}/cv-search/params-summary.pkl" \
  --train "${DATA_DIR}/PHC.feather" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --cutoffs 180 270 365 \
  --n_splits 10 \
  --output "${RESULTS_DIR}/phc-retrain-results/ours.pkl" \
  --seed 42 \
  --bootstrap_iters 1000

python validate.py \
  --mode train-performance \
  --models baecker \
  --train "${DATA_DIR}/PHC.feather" \
  --train_demographics "${DATA_DIR}/PHC-demo.feather" \
  --features "${DATA_DIR}/baecker-diagnoses.pkl" \
  --cutoffs 180 270 365 \
  --n_splits 10 \
  --output "${RESULTS_DIR}/phc-retrain-results/baecker.pkl" \
  --seed 42 \
  --bootstrap_iters 1000


### Produce analysis outputs ####
mkdir -p "${ANALYSIS_DIR}"

for input_folder in "bidmc-train-results" "bidmc-test-results" "phc-test-results" "phc-retrain-results"
do
  results_folder="${RESULTS_DIR}/${input_folder}"
  analysis_folder="${ANALYSIS_DIR}/${input_folder}"
  mkdir -p ${analysis_folder}

  echo "Run analysis for results in ${results_folder} -> ${analysis_folder}"
  python results_analysis.py \
    --mode all \
    --input ${results_folder}/*.pkl \
    --output ${analysis_folder}
  done

# Table that combines all AUCs
python results_analysis.py \
--mode auc-summary \
--auc \
  ${ANALYSIS_DIR}/bidmc-test-results/roc_auc.csv \
  ${ANALYSIS_DIR}/phc-test-results/roc_auc.csv \
  ${ANALYSIS_DIR}/phc-retrain-results/roc_auc.csv \
--name "BIDMC-Test" "PHC-Test" "PHC-Retrained" \
--output ${ANALYSIS_DIR}/auc_summary

# create combined AUC plot with CI bars
Rscript R/combined_auc_plot.R \
  --input_path ${ANALYSIS_DIR}/auc_summary-raw.csv \
  --output_path ${ANALYSIS_DIR}/auc_summary.pdf


### Model features ###
mkdir -p "${ANALYSIS_DIR}/model-weights/"

python model_weights.py \
  --model lr \
  --cutoffs 180 270 365 \
  --trained_models "${RESULTS_DIR}/bidmc-full-trained-models/ours.pkl" \
  --features "${DATA_DIR}/BIDMC-train-diagnoses.pkl" \
  --feature_desc "${ANALYSIS_DIR}/model-weights/feature_desc_map.pkl" \
  --output "${ANALYSIS_DIR}/model-weights/"


### Population summary tables for paper ###
mkdir -p "${ANALYSIS_DIR}/population-summary/"

python population_summary_table.py \
  --demographics \
    "${DATA_DIR}/BIDMC-demo-matched.feather" \
    "${DATA_DIR}/PHC-demo.feather" \
  --diagnoses \
    "${DATA_DIR}/BIDMC-matched.feather" \
    "${DATA_DIR}/PHC.feather" \
  --add_pct \
  --column_prefix BIDMC PHC \
  --output "${ANALYSIS_DIR}/population-summary/counts.tex"


### Statistical significance tests for differences in AUC
### We compute these on the BIDMC test dataset
### and on PHC (when transfering from BIDMC directly)
### We *do not* compute these again when retraining
### As this would require a large number of
### multiple comparison, not clear these would
### be meaningful to interpret
mkdir -p ${ANALYSIS_DIR}/delong-results/
# BIDMC test
python construct_proc_dataset.py \
    --input  "${RESULTS_DIR}/bidmc-test-results/ours.pkl" \
            "${RESULTS_DIR}/bidmc-test-results/baecker.pkl" \
    --output "${RESULTS_DIR}/bidmc-test-results/pROC_data.csv"

Rscript R/proc_analysis.R \
    --input "${RESULTS_DIR}/bidmc-test-results/pROC_data.csv" \
    --output "${ANALYSIS_DIR}/delong-results/bidmc_test"


# PHC test
python construct_proc_dataset.py \
    --input  "${RESULTS_DIR}/phc-test-results/ours.pkl" \
            "${RESULTS_DIR}/phc-test-results/baecker.pkl" \
    --output "${RESULTS_DIR}/phc-test-results/pROC_data.csv"

Rscript R/proc_analysis.R \
    --input "${RESULTS_DIR}/phc-test-results/pROC_data.csv" \
    --output "${ANALYSIS_DIR}/delong-results/phc_test"


### Odds (i.e. risk score) plots
python plot_odds.py \
    --input  "${RESULTS_DIR}/bidmc-test-results/pROC_data.csv" \
    --model lr \
    --cutoff 5 \
    --output "${ANALYSIS_DIR}/bidmc-test-results/odds_plot.pdf"


python plot_odds.py \
    --input  "${RESULTS_DIR}/phc-test-results/pROC_data.csv" \
    --model lr \
    --cutoff 5 \
    --output "${ANALYSIS_DIR}/phc-test-results/odds_plot.pdf"


### Supplementary tables
mkdir -p "${ANALYSIS_DIR}/bidmc-supplementary/"
python time_and_code_distributions.py \
  --input ${DATA_DIR}/BIDMC.feather \
  --output_dir "${ANALYSIS_DIR}/bidmc-supplementary/"

mkdir -p "${ANALYSIS_DIR}/phc-supplementary/"
python time_and_code_distributions.py \
  --input ${DATA_DIR}/PHC.feather \
  --output_dir "${ANALYSIS_DIR}/phc-supplementary/"



### Risk stratification
mkdir -p "${ANALYSIS_DIR}/risk-stratification/"
python risk_stratification.py \
  --train "${RESULTS_DIR}/bidmc-test-results/pROC_data.csv" \
  --test "${RESULTS_DIR}/phc-test-results/pROC_data.csv" \
  --model lr \
  --percentiles 75 99 \
  --groups "Low" "Intermediate" "High"\
  --output "${ANALYSIS_DIR}/risk-stratification/groups.csv"


# compute bootstrapped plots/values
Rscript R/risk_stratification.R \
    --input "${ANALYSIS_DIR}/risk-stratification/groups.csv" \
    --R 1000 \
    --conf 0.95 \
    --comparisons "Low-Intermediate:Intermediate-High"\
    --output_dir "${ANALYSIS_DIR}/risk-stratification"

python risk_stratification_table.py \
    --input "${ANALYSIS_DIR}/risk-stratification/summary.csv" \
    --output "${ANALYSIS_DIR}/risk-stratification/summary.tex"


### High-risk patient analysis ###
### Based on reviewer feedback we want to look at
## patients that were labeled as high-risk, then see
## if correct/incorrect prediction and what diagnoses
## they had that would have potentially led to the labeling
## (i.e. was it still reasonable?)

mkdir -p ${ANALYSIS_DIR}/risk-stratification-deep-dive/
python analyze_high_risk_labels.py \
  --risk_train_path "${RESULTS_DIR}/bidmc-test-results/pROC_data.csv" \
  --risk_test_path "${RESULTS_DIR}/phc-test-results/pROC_data.csv" \
  --risk_percentiles 75 99 \
  --risk_group_names "Low" "Intermediate" "High"\
  --model_name lr \
  --cutoff 180 \
  --diagnoses "${DATA_DIR}/PHC.feather" \
  --model_weights "${ANALYSIS_DIR}/model-weights/lr-180.csv" \
  --output "${ANALYSIS_DIR}/risk-stratification-deep-dive/high-risk-180.csv"
