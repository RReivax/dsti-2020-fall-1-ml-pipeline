import pandas as pd


def normalizeFields(input_df):

    # Format the number of installs
    input_df["Installs"] = input_df["Installs"].map(
        lambda x: x.strip()
                   .replace('+', '')
                   .replace(',', '')
    ).astype(int)

    # Format the size

    return input_df

