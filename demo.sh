#!/bin/bash
# Project Vertex v4 — demo launcher
# Starts with arc_reactor + dna_helix pre-loaded for showpiece mode.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
./run.sh --demo "$@"
