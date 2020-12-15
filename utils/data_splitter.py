 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random

# Configs
VAL_SIZE = 0.10
KEEP_HARD = False
FILENAME_SUFFIX = 'em'

# --- Get Data ----------------------------------------------------------------

# Download data
!wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -q -O hotpot_train_v1.1.json

# Import Data
with open('hotpot_train_v1.1.json') as f:
    data = json.load(f)

# --- Split Data --------------------------------------------------------------

# Divide data
data_easy = []
data_medium = []
data_hard = []

for i in data:
    if i['level'] == 'easy':
        data_easy.append(i)
    if i['level'] == 'medium':
        data_medium.append(i)
    if i['level'] == 'hard':
        data_hard.append(i)

# Set seed
random.seed(1)

# Get val indices
val_idx_easy = random.sample(range(len(data_easy)), int(len(data_easy)*VAL_SIZE))
val_idx_medium = random.sample(range(len(data_medium)), int(len(data_medium)*VAL_SIZE))
val_idx_hard = random.sample(range(len(data_hard)), int(len(data_hard)*VAL_SIZE))

# Validation splits
val_easy = [data_easy[i] for i in val_idx_easy]
val_medium = [data_medium[i] for i in val_idx_medium]
val_hard = [data_hard[i] for i in val_idx_hard] 

# Training splits
train_easy = [data_easy[i] for i in range(len(data_easy)) if i not in val_idx_easy]
train_medium = [data_medium[i] for i in range(len(data_medium)) if i not in val_idx_medium]
train_hard = [data_hard[i] for i in range(len(data_hard)) if i not in val_idx_hard]

# Validation set
if KEEP_HARD:
    train_data = train_easy + train_medium + train_hard
    val_data = val_easy + val_medium + val_hard
else:
    train_data = train_easy + train_medium
    val_data = val_easy + val_medium    

# --- Preprocessing -----------------------------------------------------------

# Function to format the context to list of strings
def context_formatter(x):
    context = []
    for i in x:
        context.append(''.join(i[1]))
    # context = ' '.join(context)
    return context

# Apply to train data
for i in train_data:
    i['context'] = context_formatter(i['context'])

# Apply to val data
for i in val_data:
    i['context'] = context_formatter(i['context'])

# --- Save Datasets -----------------------------------------------------------

with open('train_'+FILENAME_SUFFIX+'.json', 'w') as fp:
    json.dump(train_data, fp)
with open('val_'+FILENAME_SUFFIX+'.json', 'w') as fp:
    json.dump(val_data, fp)
