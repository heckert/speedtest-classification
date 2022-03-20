import pytest
import pandas as pd

from src.utils.preprocessing import parse_filters


@pytest.fixture
def df():
    df = pd.DataFrame({
        'A': list('abcd'),
        'B': range(4)
    })

    return df


def test_get_filter1(df):
    filters = {
        'A': {'equals': list('abc')},
        'B': {'greater-or-equal': [1]}
    }

    correct_result = pd.DataFrame({
        'A': list('bc'),
        'B': range(1, 3)
    }, index=[1, 2])

    filtered_df = df[parse_filters(df, filters)]

    assert filtered_df.equals(correct_result)


def test_get_filter2(df):
    filters = {
        'A': {'equals': list('abc')}
    }

    correct_result = pd.DataFrame({
        'A': list('abc'),
        'B': range(3)
    }, index=range(3))

    filtered_df = df[parse_filters(df, filters)]

    assert filtered_df.equals(correct_result)


def test_get_filter3(df):
    filters = {
        'A': {'equals': list('abc')},
        'B': {'less': [1], 'greater': [1]}
    }

    correct_result = pd.DataFrame({
        'A': list('ac'),
        'B': [0, 2]
    }, index=[0, 2])

    filtered_df = df[parse_filters(df, filters)]

    assert filtered_df.equals(correct_result)


def test_get_filter_error(df):
    filters = {
        'A': {'eq': list('abc')}
    }

    with pytest.raises(ValueError):
        parse_filters(df, filters)
