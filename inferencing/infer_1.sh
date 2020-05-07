#!/bin/bash
SECONDS=0
# ./reg_go_infer.sh G17.input00 /lambda_stor/data/brettin/ML-models/V5_docking_data_april_24/DIR.ml.3CLPRO_1.dsc.csv.reg.csv/reg_go.autosave.model.h5 /homes/hsyoo/projects/Covid19/ML-training-inferencing/descriptor_headers.csv /homes/hsyoo/projects/Covid19/ML-training-inferencing/training_headers.csv /homes/hsyoo/projects/Covid19/ML-training-inferencing/inferencing/DIR.G17/DIR.ml.3CLPRO_1.dsc.csv.reg.csv
python reg_go_infer_batch.py --in G17.input00 --model /lambda_stor/data/brettin/ML-models/V5_docking_data_april_24/DIR.ml.3CLPRO_1.dsc.csv.reg.csv/reg_go.autosave.model.h5 --dh /homes/hsyoo/projects/Covid19/ML-training-inferencing/descriptor_headers.csv --th /homes/hsyoo/projects/Covid19/ML-training-inferencing/training_headers.csv --out Infer_G17
duration=$SECONDS
echo "$duration seconds elapsed."
