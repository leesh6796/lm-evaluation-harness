#!/bin/bash
delta_layer_id=(0 3 7 11 15 19 23 27 31)
for index in ${!delta_layer_id[*]}
do
    ./custom.sh copa 20 ${delta_layer_id[index]} > copa_layer${delta_layer_id[index]}.txt
    ./custom.sh hellaswag 20 ${delta_layer_id[index]} > hellaswag_layer${delta_layer_id[index]}.txt
done
# ./custom.sh glue 20 > glue.txt
