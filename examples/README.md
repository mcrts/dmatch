## Example Description
in this example, we perform the terminology alignment of the mimic3 database against itself.

The source dataset is extracted into index format, terms are then filtered and finally we split the measurements into train and test datasets.

For each set of data, we regenarate new measurements using kernel density resampling and separate them into A and B sets.

Then, we compute the terminology alignement of sets A and B, and compute the distribution based features.

Finally, we train several matching model on the mimic3-mimic3_train set and evaluate them on the mimic3-mimic3_test set.

## Data Preparation
Extraction, filter and train test split.
```console
dmatch-tools index MIMIC3 mimic3
dmatch-tools preprocess \
    mimic3 
    --filter resources/mimic3filter.csv
dmatch-tools train_test_split mimic3
```

Generating new data sources samples (A & B) based on observations.
```console
dmatch-tools resample mimic3_test mimic3_test_A
dmatch-tools resample mimic3_test mimic3_test_B
dmatch-tools resample mimic3_train mimic3_test_A
dmatch-tools resample mimic3_train mimic3_test_B

dmatch-tools preprocess mimic3_test_A
dmatch-tools preprocess mimic3_train_A
dmatch-tools preprocess mimic3_test_B
dmatch-tools preprocess mimic3_train_B
```

Computing all pairs between train and test sets
```console
dmatch-tools prepare mimic3_test_A mimic3_test_B mimic3-mimic3_test
dmatch-tools prepare mimic3_train_A mimic3_train_B mimic3-mimic3_train
```

Computing features
```console
dmatch-tools score mimic3-mimic3_test
dmatch-tools score mimic3-mimic3_train
```

## Training model
The notebook [mimic3-mimic3.ipynb](mimic3-mimic3.ipynb) describes the training and evaluating steps involving data from mimic3-mimic3_train and mimic3-mimic3_test sets.