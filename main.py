import polars as pl
df = pl.read_parquet("data/processed/options_surfaces_data_cleaned.parquet")
print(df.columns)