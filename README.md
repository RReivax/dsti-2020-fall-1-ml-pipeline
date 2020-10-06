# dsti-2020-fall-1-ml-pipeline

Basic folder structure example:

```
.
├── app_cleaner
│   ├── Dockerfile
│   └── src
│       └── cleaner
│           ├── core
│           │   ├── format.py
│           │   └── __init__.py
│           ├── __init__.py
│           └── test
│               ├── __init__.py
│               └── test_cleaner.py
├── app_mlflow
│   └── Dockerfile
├── app_model
│   ├── Dockerfile
│   └── src
│       ├── model
│       │   ├── __init__.py
│       │   └── train.py
│       └── test
│           ├── __init__.py
│           └── test_train.py
├── clean_data
├── docker-compose.yml
├── LICENSE
├── model_data
│   └── train_output
├── raw_data
│   ├── googleplaystore.csv
│   ├── googleplaystore_user_reviews.csv
│   └── license.txt
└── README.md
```

# To try

Run the unit tests for the `app_cleaner`:
1) Build
2) Run
```bash
docker-compose build cleaner && docker-compose run cleaner python -m unittest
```