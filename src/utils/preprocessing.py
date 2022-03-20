import pandas as pd
import operator


valid_operators = {
    'equals': operator.eq,
    'greater': operator.gt,
    'greater-or-equal': operator.ge,
    'less': operator.lt,
    'less-or-equal': operator.le
}


def parse_filters(df: pd.DataFrame, filters: dict) -> pd.Series:
    """Get filters from config and create single pandas mask

    Args:
        df (pandas.DataFrame): Dataframe to be filtered.
        filters (dict): Which filters to apply.

    Returns:
        pandas.Series: Mask to filter dataframe.
    """

    # Evaluate operators passed in filters-dict
    passed_operators = set()
    operator_dicts = filters.values()
    for operator_dict in operator_dicts:
        for key in operator_dict.keys():
            passed_operators.add(key)

    allowed_operators = set(valid_operators.keys())
    intersect_allowed = set.intersection(passed_operators, allowed_operators)

    if intersect_allowed < passed_operators:
        raise ValueError('Keys for `filters` must be in {}'.format(
            ', '.join(allowed_operators))
        )

    # Iterate through passed filters and create
    # single pandas mask
    trues = pd.Series([True for _ in range(df.shape[0])])
    falses = pd.Series([False for _ in range(df.shape[0])])

    ands = trues.copy()
    for colname, operator_dict in filters.items():
        ors = falses.copy()
        for op, values in operator_dict.items():
            for value in values:
                comparison_func = valid_operators.get(op)
                tmp_cond = comparison_func(df[colname], value)

                ors = ors | tmp_cond

        ands = ands & ors

    return ands
