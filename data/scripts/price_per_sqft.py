"""Script to produce data of price per square feet"""

import os
import pandas as pd


def main():
    data = pd.read_csv('data/kc_house_data.csv')
    data['price_per_sqft'] = data['price'] / data['sqft_living']
    # Create directory "data/processed"
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/kc_house_data_per_sqft.csv', index=False)

    print(data['price_per_sqft'].max())


if __name__ == '__main__':
    main()
