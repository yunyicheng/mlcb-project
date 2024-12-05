#!/bin/bash

json_file="config.json"

if [[ -n "$SLURM_JOB_ID" || -n "$PBS_JOBID" || -d "/etc/slurm" ]]; then
  module load python/3.10.11
  echo "Python module loaded"
fi

if [ -f "$json_file" ]; then
  rm "$json_file"
fi

current_folder=$(pwd)

cat <<EOF >"$json_file"
{
  "home_folder": "$current_folder"
}
EOF

echo "config file setted up!"
mkdir -p Data
mkdir -p Results

if [ -d "pisces_env" ]; then
  source pisces_env/bin/activate
  echo "Virtual environment activated"
else
  echo "Warning: Virtual environment 'pisces_env' not found."
  python3 -m venv pisces_env
  source pisces_env/bin/activate
  echo "Virtual environment created."
fi
if [ -f "requirements.txt" ]; then
  echo "requirements.txt found. Installing dependencies..."
  pip install -r requirements.txt
  echo "Dependencies installed successfully."

else
  echo "Error: requirements.txt not found in the current directory."
  exit 1
fi
