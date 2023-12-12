# -*- coding: utf-8 -*-
"""
This script contains functions to clean population, age, and income data for every year.
"""

import pandas as pd

# set year here
year = 2020
sex_and_age_filename = f"ACSDT5Y{year}.B01001-Data.csv"

# load the ACS age data
sex_and_age_2020_raw_manhattan = pd.read_csv(sex_and_age_filename)
sex_and_age_2020_raw_brooklyn = pd.read_csv(
    f"../bk_sex_and_age_block_group_level_2021_to_13/{sex_and_age_filename}"
)
sex_and_age_2020_raw_queens = pd.read_csv(
    f"../queens_sex_by_age/{sex_and_age_filename}"
)
sex_and_age_2020_raw_bronx = pd.read_csv(f"../bronx_sex_by_age/{sex_and_age_filename}")
sex_and_age_2020_raw_staten = pd.read_csv(
    f"../staten_sex_by_age/{sex_and_age_filename}"
)
concat_df = pd.concat(
    [
        sex_and_age_2020_raw_manhattan,
        sex_and_age_2020_raw_brooklyn,
        sex_and_age_2020_raw_queens,
        sex_and_age_2020_raw_bronx,
        sex_and_age_2020_raw_staten,
    ],
    ignore_index=True,
)

# clean the dataframe
concat_df.columns = concat_df.iloc[0]
concat_df = concat_df[1:]
concat_df.reset_index(drop=True, inplace=True)
concat_df_filt = concat_df.filter(like="Estimate", axis=1)
concat_df_filt["GEO_ID"] = concat_df["Geography"]
concat_df_filt["GEO_ID"] = concat_df_filt["GEO_ID"].str[-12:]

income_filename = f"ACSDT5Y{year}.B19001-Data.csv"

# load the ACS income data(change the year in the filename accordingly)
manhattan = pd.read_csv(f"for_income/2021_to_13/{income_filename}")
brooklyn = pd.read_csv(f"for_income/bk_2021_to_13/{income_filename}")
queens = pd.read_csv(f"for_income/queens_2021_to_13/{income_filename}")
bronx = pd.read_csv(f"for_income/bronx_2021_to_13/{income_filename}")
staten = pd.read_csv(f"for_income/staten_2021_to_13/{income_filename}")

# clean the dataframe
income_df = pd.concat([manhattan, brooklyn, queens, bronx, staten], ignore_index=True)
income_df.columns = income_df.iloc[0]
income_df = income_df[1:]
income_df.reset_index(drop=True, inplace=True)
income_df_filt = income_df.filter(like="Estimate", axis=1)

# compute mean income

for col in income_df_filt.columns:
    income_df_filt[col] = pd.to_numeric(income_df_filt[col], errors="coerce")

income_df_filt_2 = income_df_filt[income_df_filt.columns].astype("float64")
income_df_filt_2 = income_df_filt_2.drop(income_df_filt_2.columns[0], axis=1)
income_values = [
    5000,
    12500,
    17500,
    22500,
    27500,
    32500,
    37500,
    42500,
    47500,
    55000,
    67500,
    85000,
    112500,
    137500,
    175000,
    200000,
]
mean_incomes = (income_df_filt_2 * income_values).sum(axis=1) / income_df_filt_2.sum(
    axis=1
)
income_df_filt_2["mean_income"] = mean_incomes

# geo id column is needed for joining with the age dataset
income_df_filt_2["GEO_ID"] = income_df["Geography"]
income_df_filt_2["GEO_ID"] = income_df_filt_2["GEO_ID"].str[-12:]

income_df_filt_final = income_df_filt_2[["mean_income", "GEO_ID"]]

income_pop_age = pd.merge(concat_df_filt, income_df_filt_final, on="GEO_ID")

# load TIGER dataset at the block group level.(change the year in the filename accordingly)
tiger_extract_2020_bg = pd.read_excel(f"../TIGER/{year}_all_boro_bg.xlsx")
tiger_extract_2020_bg.rename(columns={"GEOID": "GEO_ID"}, inplace=True)
tiger_extract_2020_bg["GEO_ID"] = tiger_extract_2020_bg["GEO_ID"].astype(str)
tiger_extract_2020_bg["GEO_ID"] = tiger_extract_2020_bg["GEO_ID"].str[-12:]
merged_df = pd.merge(income_pop_age, tiger_extract_2020_bg, on="GEO_ID")

# hardcoded value of year, change it while running the notebook
merged_df["Year"] = year

# compute male mean age
# remove the ':' from the substring for 2013-18
ages = [
    2.5,
    7,
    12,
    16,
    18.5,
    20,
    21,
    23,
    27,
    32,
    37,
    42,
    47,
    52,
    57,
    60.5,
    63,
    65.5,
    68,
    72,
    77,
    82,
    85,
]

substring = "Estimate!!Total:!!Male:!!"
filtered_columns_male = [col for col in merged_df.columns if substring in col]
merged_df_male_ages = merged_df[filtered_columns_male].astype("float64")
non_zero_rows = merged_df_male_ages.sum(axis=1) != 0
df_filtered = merged_df_male_ages[non_zero_rows]
mean_ages = (df_filtered * ages).sum(axis=1) / df_filtered.sum(axis=1)
merged_df["male_mean_age"] = mean_ages
merged_df["total_male_pop"] = df_filtered.sum(axis=1)

# compute female mean age
# remove the ':' from the substring for 2013-18
filter_string_female = "Estimate!!Total:!!Female:!!"
filtered_columns_female = [
    col for col in merged_df.columns if filter_string_female in col
]
merged_df_female_ages = merged_df[filtered_columns_female].astype("float64")
non_zero_rows_female = merged_df_female_ages.sum(axis=1) != 0
df_filtered_female = merged_df_female_ages[non_zero_rows_female]
mean_ages_female = (df_filtered_female * ages).sum(axis=1) / df_filtered_female.sum(
    axis=1
)
merged_df["female_mean_age"] = mean_ages_female
merged_df["total_female_pop"] = df_filtered_female.sum(axis=1)

# compute total population and overall mean age
merged_df["total_pop"] = df_filtered_female.sum(axis=1) + df_filtered.sum(axis=1)
merged_df["mean_age"] = (
    (df_filtered_female * ages).sum(axis=1) + (df_filtered * ages).sum(axis=1)
) / (df_filtered.sum(axis=1) + df_filtered_female.sum(axis=1))

desired_columns = [
    "GEO_ID",
    "STATEFP",
    "COUNTYFP",
    "TRACTCE",
    "BLKGRPCE",
    "ALAND",
    "INTPTLAT",
    "INTPTLON",
    "Year",
    "male_mean_age",
    "female_mean_age",
    "mean_age",
    "total_male_pop",
    "total_female_pop",
    "total_pop",
    "mean_income",
]

merged_df_final = merged_df[desired_columns]

# load TIGER block level data (change variable name accordingly - TIGER block level data is available only for the 2010 and 2020 census)
# use 2010 data for 2013-19 and 2020 data for 2020 and 2021
block_2015 = pd.read_excel("../TIGER/2020_all_boro_block.xlsx")

# joined_2021 = pd.merge(block_2015, merged_df_final, left_on=['COUNTYFP10', 'TRACTCE10'], right_on=['COUNTYFP', 'TRACTCE'])
joined_2021 = pd.merge(
    block_2015,
    merged_df_final,
    left_on=["COUNTYFP20", "TRACTCE20"],
    right_on=["COUNTYFP", "TRACTCE"],
)

# use the first line for 2013-19, second line for 2020 and 21

# final_1 = joined_2021.drop_duplicates(subset='GEOID10', keep='first')
final_1 = joined_2021.drop_duplicates(subset="GEOID20", keep="first")

final_1.rename(
    columns={
        "total_pop": f"pop_{year}",
        "mean_age": f"mean_age_{year}",
        "mean_income": f"mean_income_{year}",
    },
    inplace=True,
)

# use the first line for 2013-19 and next line for 2020-21
# final_result = final_1[['COUNTYFP10','TRACTCE10','BLOCKCE10','ALAND10','INTPTLAT10','INTPTLON10',f'pop_{year}',f'mean_age_{year}',f'mean_income_{year}']]
final_result = final_1[
    [
        "COUNTYFP20",
        "TRACTCE20",
        "BLOCKCE20",
        "ALAND20",
        "INTPTLAT20",
        "INTPTLON20",
        "GEOID20",
        f"pop_{year}",
        f"mean_age_{year}",
        f"mean_income_{year}",
    ]
]

final_result.to_excel(
    f"../6_dec_all_counties/pop_age_income_test_{year}.xlsx", index=False
)
