#!/bin/bash

python -m pip install --upgrade pip

while read requirement; do
    python -m pip install --upgrade "$requirement"
done < requirements.txt
