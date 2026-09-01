"""Tests for column bookkeeping and spectrum utilities in raman_analysis.data."""

import pandas as pd
import pytest

from raman_analysis.data import (
    is_wavenumber_column,
    report_spectrum_minimum,
    split_meta_and_spectral_columns,
)


@pytest.mark.parametrize(
    "column_name, expected",
    [
        ("500.8563", True),
        ("2999.236", True),
        ("Oil_Type", False),
        ("Chips Type", False),
        ("", False),
    ],
)
def test_is_wavenumber_column(column_name, expected):
    assert is_wavenumber_column(column_name) is expected


def test_split_meta_and_spectral_columns_sorts_spectral_ascending():
    # Deliberately out of order and interleaved with metadata, mirroring
    # how the raw CSVs are laid out (metadata first, spectra in whatever
    # order the source file happened to use).
    columns = ["Oil Type", "Chips Type", "20.5", "5.1", "10.0"]

    meta_columns, spectral_columns = split_meta_and_spectral_columns(columns)

    assert meta_columns == ["Oil Type", "Chips Type"]
    assert spectral_columns == ["5.1", "10.0", "20.5"]


def test_report_spectrum_minimum_finds_the_right_cell(capsys):
    spectra = pd.DataFrame(
        {"10.0": [1.0, -5.0], "20.0": [0.5, 2.0]}, index=[0, 1]
    )

    min_value, row_idx, col_idx = report_spectrum_minimum(spectra)

    assert min_value == -5.0
    assert row_idx == 1
    assert col_idx == "10.0"
    assert "Most negative value: -5.0" in capsys.readouterr().out
