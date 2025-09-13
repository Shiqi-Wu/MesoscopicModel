#!/bin/bash

# Get list of screen session IDs (first column)
sessions=$(screen -ls | awk '/Detached/ {print $1}')

for s in $sessions; do
    # Extract only the name part (after the dot)
    name=${s#*.}
    if [[ $name == gaussian* || $name == nearest* ]]; then
        echo "Killing screen session: $s"
        screen -S "$s" -X quit
    fi
done
