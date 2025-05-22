# model-training documentation!

## Description

Sentiment analysis

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://remla-group-5-unique-bucket/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://remla-group-5-unique-bucket/data/` to `data/`.


