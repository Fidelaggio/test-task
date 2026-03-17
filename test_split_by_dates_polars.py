import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from split_by_dates_polars import split_by_dates


@pytest.fixture
def sample_df():
    dfs = pd.date_range(
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:05",
        freq="s"
    )
    df = pd.DataFrame({"dt": dfs})
    return df


def test_empty_dataframe():
    df = pd.DataFrame(columns=["dt"])
    chunks = list(split_by_dates(df, target_size=5))
    assert len(chunks) == 0


def test_single_row():
    df = pd.DataFrame({"dt": ["2023-01-01 00:00:01"]})

    chunks = list(split_by_dates(df, target_size=2))

    assert len(chunks) == 1
    assert_frame_equal(chunks[0], df)


def test_single_chunk(sample_df):
    chunks = list(split_by_dates(sample_df, target_size=6))

    assert len(chunks) == 1
    assert_frame_equal(chunks[0], sample_df)


def test_all_same_date():
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2023-01-01 00:00:01"] * 25)
    })

    chunks = list(split_by_dates(df, target_size=7))

    assert len(chunks) == 1


@pytest.mark.parametrize("size", [1,2,4,5])
def test_all_different_dates(sample_df,size):
    import math

    l_target_size = size

    chunks = list(split_by_dates(sample_df, target_size=l_target_size))

    assert len(chunks) == math.ceil(len(sample_df) / l_target_size)
    for chunk in chunks:
        assert len(chunk) == l_target_size or (len(chunk) == 1 and chunk["dt"].nunique() == 1)


def test_normal_case_size_2():
    dfs = pd.date_range(
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:05",
        freq="s"
    )
    df = pd.DataFrame({"dt": dfs.repeat([3, 2, 4, 1, 1])})

    chunks = list(split_by_dates(df, target_size=2))

    lengths = [len(c) for c in chunks]
    assert lengths == [3, 2, 4, 2]


def test_normal_case_size_3():
    dfs = pd.date_range(
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:05",
        freq="s"
    )
    df = pd.DataFrame({"dt": dfs.repeat([3, 2, 4, 1, 1])})

    chunks = list(split_by_dates(df, target_size=3))

    lengths = [len(c) for c in chunks]
    assert lengths == [3, 6, 2]


def test_chunk_ends_exactly_on_date_change():
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2023-01-01"] * 5 + ["2023-01-02"] * 5 + ["2023-01-03"] * 5)
    })

    chunks = list(split_by_dates(df, target_size=5))

    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) == 5
        assert chunk["dt"].nunique() == 1


def test_last_chunk_smaller_than_target():
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2025-02-01"] * 8 + ["2025-02-02"] * 2)
    })

    chunks = list(split_by_dates(df, target_size=5))

    assert len(chunks) == 2
    assert [len(c) for c in chunks] == [8, 2]