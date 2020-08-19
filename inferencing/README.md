## Running on one node on the lambda system

run_all_models_lambda.sh arg1 arg2 arg3

`nohup ../../inferencing/run_all_models_2.sh 8 ORD.input .`

arg1 - number of GPUs needed to get the job done (typically 8).

arg2 - input file containing 8 filenames. Each named file (8 in total) contains a set
       feather files to run on.
       
       for example:
          $ cat ORD.input
          ORD.input00
          ORD.input01
          ORD.input02
          ORD.input03
          ORD.input04
          ORD.input05
          ORD.input06
          ORD.input07
          
          $ cat ORD.input06
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_054.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_055.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_056.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_057.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_058.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_059.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_060.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_061.feather
          /lambda_stor/data/hsyoo/descriptors/ORD/ord_062.feather

arg3 - directory that contains the model directories


reg_go_infer_dg.py
usage: reg_go_infer_batch.py [-h] [--in IN] [--model MODEL] [--dh DH]
                             [--th TH] [--out OUT]

--in filename   The named file contains the pathnames of the database shards.
                This is ORD.input06 in the example above.
                
--out dirname   This is dir where the labels and prediction files are written.
                A new directory is needed for every --in filename.
