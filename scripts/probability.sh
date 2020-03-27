#!/bin/bash

#too many arguments, best to be safe rather than sorry
if [ $# -ge 3 ]; then
    echo "Too many arguments provided! Please try again. Specify only the input file (optionally followed by the output file)."
    exit 0
fi

filename=$1
newfile=$2
#if neither file is specified, request both from command line
if [ -z $filename ] && [ -z $newfile ]; then
    #input file
    echo "Please specify an input .csv file in /data/, or type exit to quit"
    read filename
    filepath=data/$filename
    if [ "$filename" = "exit" ]; then
        exit 0
    #check file exists
    elif [ ! -f "$filepath" ]; then
        echo "/$filepath does not exist."
        scriptloc=$(readlink -f "$0")
        exec "$scriptloc"
    fi
    #output file
    echo "Please specify desired results file name, press Enter to append to the input file, or type exit to quit"
    read newfile
    if [ "$newfile" = "exit" ]; then
        exit 0
    fi
fi
#now we definitely have all our arguments assigned.

#if there is one argument it is assumed to be the input file, and results will append
#check file exists [possibly again, if nothing was specified from command line. Will try to work around this repetition]. 
filepath=data/$filename
if [ -f "$filepath" ] ; then
    #run the python code using relevant arguments
    if [ -z $newfile ]; then
        echo "No output file specified. Results will be appended to /$filepath"
        python ./py/pqrun.py $filename
    else
        echo "Output file specified. Results (+ original input) will be written to /data/$newfile"
        python ./py/pqrun.py $filename $newfile
    fi
else #restart the script if an invalid option is given
    echo "/$filepath does not exist."
    scriptloc=$(readlink -f "$0")
    exec "$scriptloc"
fi

echo "This code was written by Rhys Barnett @ Imperial College London, Feb 2020"
