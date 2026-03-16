import pandas as pd


def split_by_dates(df: pd.DataFrame, target_size: int, dt_col: str = "dt"):
    '''
    Метод работает только с заранее отсортированным DataFrame
    '''

    if df.empty:
        return

    dt = df[dt_col].to_numpy()

    start = 0
    size = 0
    prev_dt = dt[0]

    for i in range(1,len(dt)):
        size += 1
        date_changed = dt[i] != prev_dt

        if size >= target_size and date_changed:
            yield df.iloc[start:i]
            start = i
            size = 0

        prev_dt = dt[i]

    if start < len(df):
        yield df.iloc[start:]