#!/bin/bash

TS=$1
echo "run on UCLMR model on synonyms with timestamp ${TS}?"
read

SYN_STANCES="../data/${TS}_baseline_stances.csv"
SYN_BODIES="../data/${TS}_baseline_bodies.csv"
FILTERED_STANCES="../data/filtered_test_stances.csv"
FILTERED_BODIES="../data/filtered_test_bodies.csv"
UCLMR_S_RESULTS="../data/uclmr_synonym_predictions.csv"
UCLMR_F_RESULTS="../data/uclmr_filtered_predictions.csv"
UCLMR_RESULTS="../data/uclmr_results.zip"

pushd uclmr
rm -f predictions_test.csv
cp $SYN_STANCES test_stances_unlabeled.csv
cp $SYN_BODIES test_bodies.csv
echo load | python pred.py
cp predictions_test.csv $UCLMR_S_RESULTS

rm -f predictions_test.csv
cp $FILTERED_STANCES test_stances_unlabeled.csv
cp $FILTERED_BODIES test_bodies.csv
echo load | python pred.py
cp predictions_test.csv $UCLMR_F_RESULTS

7za a $UCLMR_RESULTS $UCLMR_S_RESULTS $UCLMR_F_RESULTS
popd
python score_csv.py -b $TS -t uclmr

curl -T $UCLMR_RESULTS https://transfer.sh/uclmr_results.zip
echo

