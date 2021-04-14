## Basic Overview

Distribution Matching is an Instance-Based Ontology Matching Technique.
Two terminologies can be align by comparing the distribution of term values.

This technique is applied to the case of HealthCare information system alignement,
where laboratory results tend to have the same distribution of values regardless of the*
geographical site of analysis.

This technique has been applied to the alignment of the MIMIC-III dataset with the DataWarehouse of the
Lille University Hospital.

## Installation
As the project is still under development we recommend an installation in editable mode.
```bash
git clone <github url> dmatch
pip install -e dmatch/
```
This will download the package source code into a dmatch directory, and install it in editable mode.

```bash
mkdir my_workspace
cd my_workspace
dmatch init .
```
This will create a workspace and initialize it.
The initialization procedure will copy two files :
_connections.cfg_ in which you can define your own connection strings.
_connectors.py in which you can develop your own query to extract data from datasources.
A _MimicConnector_ class is provided as a working example.

## Resources
The resources directory contain several files :
_mimic3filter.csv_ a filter file to only keep a specific set of terms.
_mimic3-mimic3reference.csv_ a reference file for training purposes.
_mimic3-mimic3.pipeline_ a model to perform distribution matching alignment.


## Command Line Interface
The package is delivered with an API _dmatch_ to perform the alignment.
```bash
dmatch align \
    MIMIC3 \
    MIMIC3 \
    resources/mimic3-mimic3.pipeline \
    mimic3-mimic3 \
     --filter1 resources/mimic3_filter.csv \
     --filter2 resources/mimic3_filter.csv \
```
This will align the MIMIC3 datasource with itself, all results will be stored in the mimic-mimic directory.
```bash
dmatch prepare \
    MIMIC3 \
    MIMIC3 \
    mimic3-mimic3 \
     --filter1 resources/mimic3_filter.csv \
     --filter2 resources/mimic3_filter.csv \
```
This will prepare a dataset of all correspondances between the two datasources.
As well as evaluate each correspondances using different measure such as
the Hellinger Distance and the Kolmogorov-Smirnov Test.
Combined with the reference file provided this will produce a dataset ready for training
a decision model.

A lower level API _dmatch-tools_ is also available for a finer control over the processus.
```bash
dmatch-tools index MIMIC3 mimic3A
dmatch-tools index MIMIC3 mimic3B
dmatch-tools preprocess mimic3A --filter resources/mimic3_filter.csv
dmatch-tools preprocess mimic3B --filter resources/mimic3_filter.csv
dmatch-tools prepare mimic3A mimic3B mimic3-mimic3
dmatch-tools score mimic3-mimic3
dmatch-tools match mimic3-mimic3 resources/mimic3-mimic3.pipeline
```