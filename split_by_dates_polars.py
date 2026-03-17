import polars as pl
import pandas as pd


def split_by_dates(df: pd.DataFrame, target_size: int, dt_col: str = "dt") -> list[pd.DataFrame]:
    '''
    Метод работает только с заранее отсортированным pandas DataFrame.
    Возвращает pandas DataFrame
    '''
    if df.empty:
        return []

    pl_df = pl.from_pandas(df)

    # group + count
    date_groups = (
        pl_df.group_by(dt_col, maintain_order=True)
        .agg(pl.len().alias("size"))
    )

    grp_ids = []
    current_grp = 0
    current_size = 0

    for row in date_groups.iter_rows(named=True):
        grp_ids.append(current_grp)
        current_size += row["size"]

        if current_size >= target_size:
            current_grp += 1
            current_size = 0

    # add col with group_id
    date_groups = date_groups.with_columns(pl.Series("grp", grp_ids))
    df_with_grp = pl_df.join(
        date_groups.select([dt_col, "grp"]),
        on=dt_col,
        how="left"
    )

    chunks = df_with_grp.partition_by("grp", maintain_order=True)
    return [chunk.drop("grp").to_pandas() for chunk in chunks]