import pandas as pd
from pathlib import Path


def get_dataset_names():
    """Get list of available datasets"""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    files = []
    for ext in ["*.csv", "*.xlsx"]:
        files.extend(datasets_dir.glob(ext))
    return [f.stem for f in files if f.is_file()]


def load_data(dataset_name):
    """Handles both CSV and Excel files automatically"""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    
    # Look for both file types
    csv_path = datasets_dir / f"{dataset_name}.csv"
    xlsx_path = datasets_dir / f"{dataset_name}.xlsx"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"No data found for {dataset_name}")