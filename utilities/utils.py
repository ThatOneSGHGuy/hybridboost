from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


def check_files_exist(
        data_path: Path, file_list: list,
) -> None:

    """
    Check if a list of files exist in the specified data path.

    Args:
        data_path (str): The path to the data directory.
        file_list (list): A list of file names to check for existence.

    Raises:
        FileNotFoundError: If any of the files do not exist.
    """
    for file in file_list:
        file_path = (data_path / file)
        if not file_path.is_file():
            error_message = f"File '{file}' does not exist."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

    logger.info(f"All specified data files exist.")


def import_csv_files(
        data_path: Path, file_list: list,
        delimiter: str = ',', decimal: str = '.', header: Union[str, int] = 'infer'
) -> Union[tuple, pd.DataFrame]:

    """
    Import a list of CSV files from the specified data path using pandas.

    Args:
        data_path (Path): The path to the data directory.
        file_list (list): A list of file names to import.
        delimiter (str, optional): The delimiter used in the CSV files (default is ',').
        decimal (str, optional): The decimal seperator used to distinguish floating part of a number
        header (str or None, optional): Whether to infer header row or None to skip it (default is 'infer').

    Returns:
        tuple: A tuple of DataFrames.
    """
    dataframes = []

    for file in file_list:
        file_path = (data_path / file)
        df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal, header=header)
        dataframes = dataframes + [df]
        logger.info(f"Data loaded from {str(file_path)}.")
        logger.info(f"Data from file {str(file)} shape: {df.shape}")

    if len(file_list) > 1:
        output_dataframes = dataframes
    else:
        output_dataframes = dataframes[0]

    return output_dataframes


def export_dataframe_to_csv(
        df: pd.DataFrame,
        output_path: Path,
        filename: str,
        delimiter: str = ',',
        index: bool = False,
        decimal: str = '.',
        header: bool = True,
) -> None:

    """
    Export a DataFrame to a CSV file with specified options.

    Args:
        df (pd.DataFrame): The DataFrame to be exported.
        output_path (Path): The path where the CSV file will be saved.
        filename (str): The name of the CSV file (without the extension).
        delimiter (str, optional): The delimiter to use between fields (default is ',').
        index (bool, optional): Whether to include the index column (default is False).
        decimal (str, optional): The character recognized as a decimal point (default is '.').
        header (bool, optional): Whether to include the column names in the output (default is True).

    Returns:
        None
    """

    # Combine the output path and filename to create the full file path
    file_path = (output_path / filename)

    try:
        # Export the DataFrame to a CSV file with specified options
        df.to_csv(file_path, sep=delimiter, index=index, decimal=decimal, header=header)
        logger.info(f"DataFrame successfully exported to {file_path}")

    except Exception as e:
        # Log an error message if the export fails
        logger.error(f"Error exporting DataFrame to Excel file: {str(e)}")


def export_dataframe_to_excel(
        df: pd.DataFrame,
        output_path: Path,
        filename: str,
        index: bool = False,
        header: bool = True,
) -> None:

    """
    Export a DataFrame to a XLSX file with specified options.

    Args:
        df (pd.DataFrame): The DataFrame to be exported.
        output_path (Path): The path where the XLSX file will be saved.
        filename (str): The name of the XLSX file (without the extension).
        index (bool, optional): Whether to include the index column (default is False).
        header (bool, optional): Whether to include the column names in the output (default is True).

    Returns:
        None
    """

    # Combine the output path and filename to create the full file path
    file_path = (output_path / filename)

    try:
        # Export the DataFrame to a CSV file with specified options
        df.to_excel(file_path, index=index, header=header)
        logger.info(f"DataFrame successfully exported to {file_path}")

    except Exception as e:
        # Log an error message if the export fails
        logger.error(f"Error exporting DataFrame to Excel file: {str(e)}")
