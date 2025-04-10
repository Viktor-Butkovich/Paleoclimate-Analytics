from typing import List
from modules import data_types
import polars as pl


def get_year_bin(year: int) -> int:
    if year < -800000:  # Nearest 50,000
        return round(year / 50000) * 50000
    elif year < -20000:  # Nearest 2000
        return round(year / 2000) * 2000
    elif year < 0:  # Nearest 250
        return round(year / 250) * 250
    elif year < 1850:  # Nearest 50
        return round(year / 50) * 50
    elif year < 2025:  # Nearest 1
        return round(year)
    else:  # Next 1000
        return round((year + 1000) / 1000) * 1000


def include_sample(sample: data_types.Temp12k_TS_sample) -> bool:
    if sample.get("paleoData_units") != "degC":
        return False
    elif sample.get("paleoData_useInGlobalTemperatureAnalysis", "TRUE") == "FALSE":
        return False
    elif "DELETE" in sample.get("paleoData_QCnotes", ""):
        return False
    elif sample.get("age", None) == None:
        return False
    elif sample.get("paleoData_values")[0] == "nan" or sample.get("age")[0] == "nan":
        return False
    else:
        return True


def year_bins_transform(df: pl.DataFrame, valid_year_bins: List[int]) -> pl.DataFrame:
    df = (
        df.with_columns(
            pl.col("year")
            .map_elements(get_year_bin, return_dtype=pl.Int64)
            .alias("year_bin")
        )
        .group_by("year_bin")
        .agg([pl.col(col).mean().alias(col) for col in df.columns if col != "year_bin"])
        .drop("year")
    )
    missing_year_bins = set(valid_year_bins) - set(df["year_bin"].unique())
    missing_entries = pl.DataFrame(
        {
            "year_bin": list(missing_year_bins),
            **{
                col: [None] * len(missing_year_bins)
                for col in df.columns
                if col != "year_bin"
            },
        }
    )

    df = pl.concat([df, missing_entries]).sort("year_bin")

    for col in df.columns:
        if col != "year_bin":
            df = df.with_columns(pl.col(col).interpolate().alias(col))

    df = df.with_columns(
        [
            pl.col(col).fill_null(strategy="backward").fill_null(strategy="forward")
            for col in df.columns
        ]
    )

    for col in [
        "co2_ppm",
        "co2_radiative_forcing",
        "anomaly",
        "be_ppm",
        "VADM",
    ]:  # Remove future filled null values
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col("year_bin") > 2025)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df
