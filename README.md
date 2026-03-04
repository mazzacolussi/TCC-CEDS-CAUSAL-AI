# tcc-causal-AI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.


## Data

For this project, the datasets from [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download)
are used. After downloading the individual datasets, make sure they are placed in the `data/raw/` folder at the root of this project.

The relationships between the datasets are shown below:

<p align="center">
  <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download">
    <img src="references/figures/datasets-flow.png" width="70%">
  </a>
</p>


To generate a unified and preprocessed dataset at the `order_id` and `customer_id` level, execute the following command:

```bash
python src/app/dataset.py
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tcc_causal_ai and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
├── setup.cfg          <- Configuration file for flake8
│
└── src 
     └── app           <- Source code for use in this project.
          │
          ├── __init__.py             <- Makes tcc_causal_ai a Python module
          │
          ├── dataset.py              <- Scripts to download or generate data
          │
          ├── config                
          │   ├── __init__.py 
          │   └── settings.py         <- Store useful variables and configuration
          │
          ├── modeling                
          │   ├── __init__.py 
          │   ├── predict.py          <- Code to run model inference with trained models          
          │   └── train.py            <- Code to train models
          │
          └── plots.py                <- Code to create visualizations
```

--------

