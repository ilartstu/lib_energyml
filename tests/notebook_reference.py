"""The SVGP notebook's cleaning (STEP 3.1-3.10), kept as close to the original as possible.

Used only to check that ``energyprime.preprocessing.Cleaner`` reproduces it.
"""
import numpy as np
import pandas as pd


def exact_zero_mask(series):
    return series.notna() & series.eq(0)


def zero_or_nan_mask(series):
    return series.isna() | exact_zero_mask(series)


def find_long_zero_nan_blocks(df, columns, min_length):
    long_block_mask_by_column = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        current_mask = zero_or_nan_mask(df[col])
        group_id = current_mask.ne(current_mask.shift(fill_value=False)).cumsum()
        for _, group_index in current_mask.groupby(group_id).groups.items():
            group_index = list(group_index)
            if not current_mask.iloc[group_index[0]]:
                continue
            if len(group_index) < min_length:
                continue
            long_block_mask_by_column.loc[group_index, col] = True
    return long_block_mask_by_column


def zscore_outlier_mask(series, threshold):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = s.mask(exact_zero_mask(s), np.nan).dropna()
    if len(clean) < 10:
        return pd.Series(False, index=series.index)
    mean_value = clean.mean()
    std_value = clean.std()
    if std_value == 0 or pd.isna(std_value):
        return pd.Series(False, index=series.index)
    z_score = (s - mean_value).abs() / std_value
    return (z_score > threshold).fillna(False)


def physical_outlier_mask(series, column_name, limits):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lower_bound, upper_bound = limits.get(column_name, (None, None))
    mask = pd.Series(False, index=series.index)
    if lower_bound is not None:
        mask = mask | (s < lower_bound)
    if upper_bound is not None:
        mask = mask | (s > upper_bound)
    return mask.fillna(False)


def interpolate_local(df, columns, outlier_mask, min_len, max_len):
    result = df.copy()
    for col in columns:
        original = result[col].copy()
        current = zero_or_nan_mask(original)
        group_id = current.ne(current.shift(fill_value=False)).cumsum()
        group_size = current.groupby(group_id).transform("size")
        local = current & (group_size >= min_len) & (group_size <= max_len)
        selected = local | outlier_mask[col]
        interpolated = original.mask(selected, np.nan).interpolate(method="linear", limit_area="inside")
        result[col] = original
        result.loc[selected, col] = interpolated.loc[selected]
    return result


def svgp_notebook_clean(data: pd.DataFrame, columns, physical_limits, long_min=25, local_min=4,
                        local_max=24, zscore_threshold=5.0):
    """Returns the cleaned frame with the time as index (the notebook keeps it in x_time)."""
    df = data.reset_index()
    time_col = df.columns[0]
    x_time = df[time_col]
    df = df[columns]
    long_mask = find_long_zero_nan_blocks(df, columns, long_min)
    delete = long_mask.any(axis=1)
    df = df.loc[~delete].reset_index(drop=True)
    x_time = x_time.loc[~delete].reset_index(drop=True)
    outliers = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        z = zscore_outlier_mask(df[col], zscore_threshold)
        phys = physical_outlier_mask(df[col], col, physical_limits)
        missing = zero_or_nan_mask(df[col])
        outliers[col] = (z & ~missing) | (phys & ~missing)
    result = interpolate_local(df, columns, outliers, local_min, local_max)
    result.index = pd.DatetimeIndex(x_time)
    return result
