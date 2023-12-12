"""
This script contains functions to process health indicators data.
"""


import pandas as pd
import geopandas as gpd

county_dict = {
    "Bronx": "Bronx County",
    "Brooklyn": "Kings County",
    "Manhattan": "New York County",
    "Queens": "Queens County",
    "Staten Island": "Richmond County",
}


def shp_to_csv(shapefile_path, csv_output_path):
    """
    Convert a shapefile to a CSV file.

    Args:
      shapefile_path (str): Path to the shapefile.
      csv_output_path (str): Path to the output CSV file.

    Returns:
      None
    """
    gdf = gpd.read_file(shapefile_path)
    gdf.to_csv(csv_output_path, index=False)


def getFIPS():
    """
    Read FIPS data from a file, process it to extract state and county codes, and filter records for the state code '36' (New York).

    Returns:
      filtered_fips (DataFrame): A pandas DataFrame containing FIPS data with columns 'FIP', 'countyName', 'stateFP', and 'countyFP'. The data is filtered to include only records for the state with the code '36' (New York).
    """
    fips = pd.read_csv(
        "fips.txt",
        sep=" " * 8,
        header=None,
        names=["FIP", "countyName"],
        skiprows=72,
        error_bad_lines=False,
    )

    fips["stateFP"] = fips["FIP"] // 1000
    fips["countyFP"] = fips["FIP"] % 1000

    fips["countyFP"] = fips["countyFP"].astype(int)

    fips["stateFP"] = fips["stateFP"].apply(lambda x: f"{x:02d}")
    fips["countyFP"] = fips["countyFP"].apply(lambda x: f"{x:03d}")

    filtered_fips = fips[fips["countyName"].isin(county_dict.values())]

    return filtered_fips


def processTiger(csv_file_path):
    """
    Process TIGER demographic data.

    Args:
      csv_file_path (str): Path to the CSV file containing TIGER demographic data.

    Returns:
      tiger (DataFrame): A pandas DataFrame containing processed TIGER demographic data.
    """
    tiger = pd.read_csv(csv_file_path)
    tiger = tiger.drop(
        columns=[
            "MTFCC20",
            "UR20",
            "UACE20",
            "UATYPE20",
            "FUNCSTAT20",
            "ALAND20",
            "AWATER20",
            "HOUSING20",
            "NAME20",
            "POP20",
            "MTFCC10",
            "UR10",
            "UACE10",
            "UATYPE",
            "FUNCSTAT10",
            "ALAND10",
            "AWATER10",
            "NAME10",
        ],
        axis=1,
        errors="ignore",
    )
    tiger = tiger.assign(
        **{"COUNTYFP10": tiger["COUNTYFP10"].apply(lambda x: f"{x:03d}")}
        if "COUNTYFP10" in tiger.columns
        else {}
    )
    tiger = tiger.assign(
        **{"COUNTYFP20": tiger["COUNTYFP20"].apply(lambda x: f"{x:03d}")}
        if "COUNTYFP20" in tiger.columns
        else {}
    )
    return tiger


def processCOVID(csv_file_path):
    """
    Process COVID hospitalization rate data.

    Args:
      csv_file_path (str): Path to the CSV file containing COVID hospitalization rate data.

    Returns:
      COVID_hospperyear (DataFrame): A pandas DataFrame containing processed COVID hospitalization rate data.
    """
    hosprate_df = pd.read_csv(csv_file_path)
    columns_to_keep = [
        "date",
        "HOSPRATE_Bronx",
        "HOSPRATE_Brooklyn",
        "HOSPRATE_Manhattan",
        "HOSPRATE_Queens",
        "HOSPRATE_Staten_Island",
    ]

    hosprate_filtered = hosprate_df[columns_to_keep]
    hosprate_filtered["date"] = pd.to_datetime(
        hosprate_filtered["date"], format="%m/%Y"
    )

    average_hospitalizations_by_year = (
        hosprate_filtered.groupby(hosprate_filtered["date"].dt.year).mean().iloc[:, :5]
    )
    average_hospitalizations_by_year = average_hospitalizations_by_year.reset_index()

    melted_df = pd.melt(
        average_hospitalizations_by_year,
        id_vars=["date"],
        value_vars=[
            "HOSPRATE_Bronx",
            "HOSPRATE_Brooklyn",
            "HOSPRATE_Manhattan",
            "HOSPRATE_Queens",
            "HOSPRATE_Staten_Island",
        ],
        var_name="countyName",
        value_name="Hosprate",
    )
    melted_df.rename(columns={"date": "year"}, inplace=True)
    melted_df = melted_df[["year", "Hosprate", "countyName"]]

    county_dict = {
        "Bronx": "Bronx County",
        "Brooklyn": "Kings County",
        "Manhattan": "New York County",
        "Queens": "Queens County",
        "Staten Island": "Richmond County",
    }
    melted_df["countyName"] = (
        melted_df["countyName"].str[9:].str.replace("_", " ").replace(county_dict)
    )

    fips = getFIPS()
    COVID_hospperyear = pd.merge(melted_df, fips, on="countyName", how="left")

    COVID_hospperyear["hosprate_2020"] = COVID_hospperyear.loc[
        COVID_hospperyear["year"] == 2020, "Hosprate"
    ]
    COVID_hospperyear["hosprate_2021"] = COVID_hospperyear.loc[
        COVID_hospperyear["year"] == 2021, "Hosprate"
    ]
    COVID_hospperyear["hosprate_2022"] = COVID_hospperyear.loc[
        COVID_hospperyear["year"] == 2022, "Hosprate"
    ]
    COVID_hospperyear["hosprate_2023"] = COVID_hospperyear.loc[
        COVID_hospperyear["year"] == 2023, "Hosprate"
    ]

    COVID_hospperyear = COVID_hospperyear.drop(
        columns=["year", "FIP", "Hosprate"], axis=1
    )

    COVID_hospperyear = COVID_hospperyear.groupby("countyFP", as_index=False).agg(
        {
            "countyName": "first",
            "stateFP": "first",
            "hosprate_2020": "sum",
            "hosprate_2021": "sum",
            "hosprate_2022": "sum",
            "hosprate_2023": "sum",
        }
    )

    COVID_hospperyear = COVID_hospperyear[
        [
            "stateFP",
            "countyFP",
            "countyName",
            "hosprate_2020",
            "hosprate_2021",
            "hosprate_2022",
            "hosprate_2023",
        ]
    ].rename(
        columns={
            "hosprate_2020": "covid_hosprate_2020",
            "hosprate_2021": "covid_hosprate_2021",
            "hosprate_2022": "covid_hosprate_2022",
            "hosprate_2023": "covid_hosprate_2023",
        }
    )

    return COVID_hospperyear


def process_CHIRS(allIndicators, countyName):
    """
    Process CHIRS data for a specific county and indicator.

    Args:
      allIndicators (DataFrame): DataFrame containing all CHIRS indicators.
      countyName (str): Name of the county for which the data should be processed.

    Returns:
      DataFrame: Processed CHIRS data for heart attack hospitalization rate per 10,000 for the specified county and indicator. Returns None if no data is found.
    """
    NYC_indicators = allIndicators[allIndicators["Geographic area"] == countyName]

    indicator = "Cardiovascular Disease Indicators"
    filtered_df = NYC_indicators[
        (NYC_indicators["Topic Area"] == indicator)
        & (
            NYC_indicators["Indicator Title"]
            == "Heart attack hospitalization rate per 10,000"
        )
        & (~NYC_indicators["Year"].str.contains("-"))
    ]

    filtered_df["Rate/Percent"] = filtered_df["Rate/Percent"].astype(float) / 10

    if not filtered_df.empty:
        HA_final = filtered_df.pivot(
            index="Geographic area", columns="Year", values="Rate/Percent"
        ).reset_index()
        HA_final.columns.name = None
        HA_final.columns = ["Geographic Area"] + [
            f"cardio_{year}" for year in HA_final.columns[1:]
        ]
        return HA_final
    else:
        return None


def processHA(csv_file_path):
    """
    Process Heart Attack (HA) data for multiple counties.

    Args:
      csv_file_path (str): Path to the CSV file containing CHIRS indicators.

    Returns:
      DataFrame: Processed Heart Attack (HA) data for multiple counties, including FIPS information.
    """
    NY_indicators = pd.read_excel(csv_file_path, sheet_name=1)

    results_list = []

    for county_name in county_dict.values():
        result = process_CHIRS(NY_indicators, county_name)
        results_list.append(result)

    HA_final = pd.concat(results_list, ignore_index=True)

    fips = getFIPS()

    HA_final = pd.merge(
        HA_final, fips, left_on="Geographic Area", right_on="countyName", how="left"
    )

    HA_final = HA_final[
        [
            "stateFP",
            "countyFP",
            "countyName",
            "cardio_2011",
            "cardio_2012",
            "cardio_2013",
            "cardio_2014",
            "cardio_2016",
            "cardio_2017",
            "cardio_2018",
            "cardio_2019",
            "cardio_2020",
        ]
    ]

    return HA_final


def mergeCOVID(tiger, processedCOVID):
    """
    Merge Tiger data with processed COVID hospitalization rate data.

    Args:
      tiger (DataFrame): TIGER demographic data DataFrame.
      processedCOVID (DataFrame): Processed COVID hospitalization rate data DataFrame.

    Returns:
      DataFrame: Merged DataFrame containing TIGER demographic data and processed COVID hospitalization rate data.
    """
    tiger_COVID_merged = pd.merge(
        tiger, processedCOVID, left_on="COUNTYFP20", right_on="countyFP", how="inner"
    )

    tiger_COVID_merged = tiger_COVID_merged.drop(
        columns=["stateFP", "countyFP"], axis=1
    )

    return tiger_COVID_merged


def mergeHA(tiger, processedHA):
    """
    Merge Tiger data with processed Heart Attack (HA) data.

    Args:
      tiger (DataFrame): TIGER demographic data DataFrame.
      processedHA (DataFrame): Processed Heart Attack (HA) data DataFrame.

    Returns:
      DataFrame: Merged DataFrame containing TIGER demographic data and processed Heart Attack (HA) data.
    """
    tiger_HA_merged = pd.merge(
        tiger, processedHA, left_on="COUNTYFP20", right_on="countyFP", how="inner"
    )

    tiger_HA_merged = tiger_HA_merged.drop(columns=["stateFP", "countyFP"], axis=1)

    return tiger_HA_merged
