# Prototype-based learning for tabular data

Implementation of prototype-based learning for tabular data. 

## Installation

1-  Get the data by runing `make get-data` (or see manual download instructions below). 


2- Install necessary packages

Create a virtual environment and install the necessary requirements, e.g. 

    python3 -m venv .env_proto
    source .env_proto/bin/activate
    pip install -r requirements.txt

(see `requirements_frozen.txt` to explicitly see all packages and exact versions). 

3- (Optional) Run the tests

    make test

## Notebooks

See the `notebooks/` for examples of usage. 

## Manual data download

The following is necessary to run tests and notebooks: 

UCI Banknote dataset: 
- Download the data from [here](https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip). 
- Unzip, rename the resulting file `banknote.txt` and move it to `data/`.

King county dataset: 

- Download the data from [here](https://geodacenter.github.io/data-and-lab/data/kingcounty.zip). 
- Unzip and move the file `kingcounty/kc_house_data.csv` to `data/`. 
- Remove everything else.

King county prices per square feet: 
- Once the previous files are downloaded, run `python data/scripts/price_per_sqft.py`
