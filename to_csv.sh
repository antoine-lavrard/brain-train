#!/bin/bash
# ./to_csv.sh miniimagenet/

#dir=$1
#csv_file=${dir##*/}.csv  # create a csv file with the same name as the directory

#!/bin/bash

# Specify the directory where the result files are located
dir=miniimagenet_LEAKY_ME/

# Specify the name of the output CSV file
output=${dir}result.csv

# Write the header row of the CSV file
echo "Model,Accuracy,Confidence" > $output

# Iterate over each result file in the directory
find $dir -type f -name "*.txt" | while read file ; do
	# Extract the model name from the file path
	model=$(echo $file | sed "s|$dir||" | cut -d'/' -f2)
    echo $file;
	# Extract the accuracy and confidence values from the last two lines of the file
	accuracy=$(tail -n 2 $file |awk {'print $1'});
	confidence=$(tail -n 2 $file |awk {'print $6'});
    echo $(tail -n 2 $file |awk {'print $1'});
	# Write the model name, accuracy, and confidence to the CSV file
	echo "$model,$accuracy,$confidence" >> $output
done