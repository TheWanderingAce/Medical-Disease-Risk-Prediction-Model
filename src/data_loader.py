from pathlib import Path
import zipfile
import pandas as pd

#Get absolute root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Set path to zip file
ZIP_PATH = PROJECT_ROOT / "dataset" / "DiabetesHealthIndicatorsDataset.zip"

#Set name of csv file used for project
CSV_NAME = "diabetes_binary_health_indicators_BRFSS2015.csv"

def load_data():
    '''
    Load the dataset directly from zip file

    Returns:
        pd.DataFrame: The diabetes dataset.
    '''

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset Zip file not found: {ZIP_PATH}"
        )

    #open ZIP archive
    with zipfile.ZipFile(ZIP_PATH) as archive:

        if CSV_NAME not in archive.namelist():
            raise FileNotFoundError(
                f"{CSV_NAME} was not found in the ZIP "
                f"Available files: {archive.namelist()}"
            )
        
            #Open the CSV inside the ZIP
        with archive.open(CSV_NAME) as csv_file:

            #Load the CSV file into pandas dataframe
            df = pd.read_csv(csv_file)

            
    return df
            

# Run this section only when executing this file directly
if __name__ == "__main__":

    df = load_data()

    print("Dataset loaded successfully!")

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())