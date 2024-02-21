#!/bin/bash

# copy relevant files to JASMIN

rootfolder=`dirname $0`/..
username=$1
destfolder=$2

if [ -z ${username} ] || [ -z ${destfolder} ];
then
  echo provide the username and destination folder on JASMIN as arguments
else
  rsync -avr --delete --exclude "*/__pycache__" $rootfolder/src $username@login2.jasmin.ac.uk:$destfolder/cae_tools
  rsync -avr --delete --exclude "*/__pycache__" $rootfolder/tools $username@login2.jasmin.ac.uk:$destfolder/cae_tools
  rsync -avr --delete $rootfolder/example_pipelines $username@login2.jasmin.ac.uk:$destfolder/cae_tools
  rsync -avr $rootfolder/pyproject.toml $username@login2.jasmin.ac.uk:$destfolder/cae_tools
  rsync -avr $rootfolder/setup.cfg $username@login2.jasmin.ac.uk:$destfolder/cae_tools
fi




