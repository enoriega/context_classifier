#!/bin/sh

for F in annotations2/PMC*
do
    ID=${F#annotations2/}

    for X in sentences.txt titles.txt citations.txt
    do
        S1=$(wc -l annotations2/$ID/$X)
        S2=$(wc -l annotations/$ID/$X)

        if [ ${S1% *} != ${S2% *} ]
        then
            echo $ID $X!
        fi
    done
done
