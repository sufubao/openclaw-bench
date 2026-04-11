#!/usr/bin/env bash

control_file=$1
openclaw-bench generate-config \
	--control-file "control/${control_file}.json" \
	--output "configs/${control_file}__$(date +%Y-%m-%d_%H-%M-%S).json"

