import pandas as pd

from dataset_utils import Dataset

if __name__ == '__main__':
    data = Dataset(years_to_target=5)
    data.create_dataset()
    print("dataset created")