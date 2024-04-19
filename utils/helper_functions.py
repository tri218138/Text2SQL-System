from pathlib import Path
import pandas as pd


def import_database():
    print("Save global schema at runs/global_schema.txt")


def get_excel_schema(_file):
    filename, extension = _file.split(".")
    with open(Path("runs") / "tables" / _file, encoding="utf-8") as file:
        if extension == "csv":
            describe = pd.load_csv(file).describe
        elif extension == "xlsx":
            describe = pd.load_xlsx(file).describe

    # process 'describe'
    return describe


def read_table_file(_file) -> pd.DataFrame:
    filename, extension = _file.split(".")
    with open(Path("runs") / "tables" / _file, encoding="utf-8") as file:
        if extension == "csv":
            df = pd.load_csv(file)
        elif extension == "xlsx":
            df = pd.load_xlsx(file)
    return df
