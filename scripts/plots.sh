#!/bin/bash

#too many arguments, best to be safe rather than sorry
if [ $# -ge 2 ]; then
    echo "Too many arguments provided! Please try again. Specify only the input file."
    exit 0
fi

filename=$1
#if no file is specified, request both from command line
if [ -z $filename ]; then
    #input file
    echo "Please specify an input .csv file in /data/, or type exit to quit"
    read filename
    filepath=./data/$filename
    if [ "$filename" = "exit" ]; then
        exit 0
    #check file exists
    elif [ ! -f "$filepath" ]; then
        echo "/$filepath does not exist."
        scriptloc=$(readlink -f "$0")
        exec "$scriptloc"
    fi
fi
#now we definitely have all our arguments assigned.

#if there is one argument it is assumed to be the input file, and results will append
#check file exists [possibly again, if nothing was specified from command line. Will try to work around this repetition]. 
filepath=./data/$filename
if [ -f "$filepath" ] ; then
    #run the python code using relevant arguments
    python ./py/plots.py $filename
else #restart the script if an invalid option is given
    echo "/$filepath does not exist."
    scriptloc=$(readlink -f "$0")
    exec "$scriptloc"
fi

