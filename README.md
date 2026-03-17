# What-Makes-a-Good-Commit-Message
This repository contains the main data and scripts used in 'What Makes a Good Commit Message'
## Dataset
The folder dataset contains the following files.
* literature survery.xlsx
    * It contains the data of 46 relevant literatures reviewed in this study (Section 3.2).

* questionnaire.csv
    * It is the questionnaire which sent to experienced contributors.
    * It contains three questions.
    * It also contains an example of the actual content of the email.
  
* posts list.xlsx
    * It contains all posts we studied in Sec. 3.2.

* sampled messages.csv
    * It contains meta-information of 1649 labeled commit messages.
    * label = 0 means a commit message contains "Why and What".
    * label = 1 means a commit message  contains "Neither Why nor What".
    * label = 2 means a commit message  contains "No What".
    * label = 3 means a commit message  contains "No Why".
    * if_mulit_commit = 1 means a commit is non-atomic.
  
* maintenance type and expression way.xlsx
    * It contains the results of our RQ2: the expression ways of Why and What, as well as links to maintenance types.
  
## CommitMessage (Scripts)
The folder scripts contains the following files.

* Preprocessor
  * It contains the preprocessing of commit message, including the replacement of token in message, etc.

* ModelTraining
  * It contains the code for our model training, that is, the implementation of different classification techniques.

## Replication

To help researchers replicate our results, we provide a simple inference script that outputs the same predictions as in the paper. The script can be run from source or with Docker.

The package outputs:
* `contains_why`
* `contains_what`
* `is_good`
* `why_score`
* `what_score`

`is_good = contains_why and contains_what`.

### Run From Source
1. Create the `uv` environment:
   ```bash
   uv sync --frozen
   ```
2. Download the reproduced checkpoints (optional):
   ```bash
   uv run python download_models.py
   ```
3. Run inference for a single commit message:
   ```bash
   uv run python replication_infer.py "add retry logic because connection may fail"
   ```
4. Run batch inference for many commit messages:
   ```bash
   uv run python replication_infer.py \
     --input-file commits.txt \
     --batch-size 32 \
     --output-file predictions.jsonl
   ```

Batch input formats:
* `.txt`: one commit message per line
* `.jsonl`: use `--text-field` to select the message field
* `.csv`: use `--text-field` to select the message column

Examples:
```bash
uv run python replication_infer.py \
  --input-file commits.jsonl \
  --text-field message \
  --batch-size 32 \
  --output-file predictions.jsonl
```

```bash
uv run python replication_infer.py \
  --input-file commits.csv \
  --text-field message \
  --batch-size 32 \
  --output-file predictions.jsonl
```

If CUDA is available, PyTorch will use GPU automatically. You can also force a device:
```bash
uv run python replication_infer.py --device cpu "fix typo in docs"
```

### Run With Docker
Build the image:
```bash
docker build -t whatgoodcm-replication:latest .
```

Run a single prediction:
```bash
docker run --rm whatgoodcm-replication:latest "add retry logic because connection may fail"
```

Run batch inference with a bind mount:
```bash
docker run --rm -v "$PWD:/data" whatgoodcm-replication:latest \
  --input-file /data/commits.txt \
  --batch-size 32 \
  --output-file /data/predictions.jsonl
```

If the host has an NVIDIA GPU and the container runtime is configured, enable GPU inference with:
```bash
docker run --rm --gpus all whatgoodcm-replication:latest "add retry logic because connection may fail"
```


