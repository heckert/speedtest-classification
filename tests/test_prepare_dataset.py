import pytest
import pandas as pd
from src.prepare_dataset import get_filters

from omegaconf import OmegaConf


@pytest.fixture
def df():
    df = pd.DataFrame({
        'A': list('abcd'),
        'B': range(4)
    })

    return df

def apply_filter(df, cfg):
    cfg = OmegaConf.create(cfg)
    filter_ = get_filters(df, cfg)
    return df[filter_]

def test_get_filter1(df):
    cfg = {
        'filters':{
            'A': list('abc'),
            'B': list(range(1,4))
        }
    }

    correct_result = pd.DataFrame({
        'A': list('bc'),
        'B': range(1,3)
    }, index=[1,2])

    assert apply_filter(df, cfg).equals(correct_result)

def test_get_filter2(df):
    cfg = {
        'filters':{
            'A': list('abc')
        }
    }

    correct_result = pd.DataFrame({
        'A': list('abc'),
        'B': range(3)
    }, index=range(3))

    assert apply_filter(df, cfg).equals(correct_result)
        

