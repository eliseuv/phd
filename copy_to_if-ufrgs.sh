#!/bin/bash

rsync -Pazvh data/ if-ufrgs:programs/phd/data/
rsync -Pazvh plots/ if-ufrgs:programs/phd/plots/
