#!/bin/bash

NAME=`echo "$1" | cut -d'.' -f1`

echo $NAME
dot $1 -Tpng -o $NAME.png
