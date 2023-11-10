#!/bin/bash

[ "$(ls -A /home/clustering_benchmarking)" ] || cp -R /home/dump/. /home/clustering_benchmarking
cd /home/clustering_benchmarking
chmod 777 ./scripts/*

python src/main.py