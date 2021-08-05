#!/usr/bin/env bash

./build.sh

docker save mixlacune | gzip -c > mixlacune.tar.gz
