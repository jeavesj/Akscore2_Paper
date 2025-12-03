import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', required=True, help='Directory where all outputs are stored in assumed structure basedir/pdbid/filename.csv')
parser.add_argument('--filename', required=True, help='Common name of all output files in assumed structure basedir/pdbid/filename.csv')
parser.add_argument('--output', required=True, help='Path to merged output file')
args = parser.parse_args()

all_preds_info = []

for pdbid in os.listdir(args.basedir):
    fpath = os.path.join(args.basedir, pdbid, args.filename)
    
    preds_info = pd.read_csv(fpath)
    preds_info['pdbid'] = pdbid
    
    all_preds_info.append(preds_info)
    
df = pd.concat(all_preds_info)
df.to_csv(args.output, index=False)
