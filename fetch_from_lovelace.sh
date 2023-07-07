#!/bin/sh

RSYNC="rsync -Pazvhm"
SRC_DIR="lovelace:programs/comp-phys/data/"
DEST_DIR="./data/magnet_ts_wishart/"

$RSYNC --include='BlumeCapelSq2DEigvals*' --include='*/' --exclude='*' $SRC_DIR $DEST_DIR/blume-capel_2d/
$RSYNC --include='BlumeCapelSq3DEigvals*' --include='*/' --exclude='*' $SRC_DIR $DEST_DIR/blume-capel_3d/

$RSYNC --include='BlumeCapelSq2DCorrelations*' --include='*/' --exclude='*' $SRC_DIR $DEST_DIR/blume-capel_2d/
$RSYNC --include='BlumeCapelSq3DCorrelations*' --include='*/' --exclude='*' $SRC_DIR $DEST_DIR/blume-capel_3d/
