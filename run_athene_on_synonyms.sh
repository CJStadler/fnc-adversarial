#!/bin/bash

TS=$1
echo "run on Athene model on synonyms with timestamp ${TS}?"
read

SYN_STANCES="../data/${TS}_baseline_stances.csv"
SYN_BODIES="../data/${TS}_baseline_bodies.csv"
FILTERED_STANCES="../data/filtered_test_stances.csv"
FILTERED_BODIES="../data/filtered_test_bodies.csv"
ATHENE_S_RESULS="../data/athene_synonym_predictions.csv"
ATHENE_F_RESULS="../data/athene_filtered_predictions.csv"
ATHENE_RESULTS="../data/athene_results.zip"

cd athene_system
rm -f data/fnc-1/fnc_results/submission.csv
cp $SYN_STANCES data/fnc-1/test_stances_unlabeled.csvv
cp $SYN_BODIES data/fnc-1/test_bodies.csv
python fnc/pipeline.py -p ftest > predict_synonyms.log
cp data/fnc-1/fnc_results/submission.csv $ATHENE_S_RESULTS

rm -f data/fnc-1/fnc_results/submission.csv
cp $FILTERED_STANCES data/fnc-1/test_stances_unlabeled.csv
cp $FILTERED_BODIES data/fnc-1/test_bodies.csv
python fnc/pipeline.py -p ftest > predict_filtered.log
cp data/fnc-1/fnc_results/submission.csv $ATHENE_F_RESULTS

7za a $ATHENE_RESULTS $ATHENE_S_RESULS $ATHENE_F_RESULS
curl -T $ATHENE_RESULTS https://transfer.sh/athene_results.zip
echo

