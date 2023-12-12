"""
This script contains functions to merge and clean population, age, and income data.
"""

import pandas as pd

df_2013 = pd.read_excel("pop_age_income_2013.xlsx")
df_2014 = pd.read_excel("pop_age_income_2014.xlsx")
df_2014_d = df_2014.drop(columns=["INTPTLAT10", "INTPTLON10", "ALAND10"])
merge_13_14 = pd.merge(df_2013, df_2014_d, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

df_2015 = pd.read_excel("pop_age_income_2015.xlsx").drop(
    columns=["INTPTLAT10", "INTPTLON10", "ALAND10"]
)

df_2016 = pd.read_excel("pop_age_income_2016.xlsx").drop(
    columns=["INTPTLAT10", "INTPTLON10", "ALAND10"]
)
df_2017 = pd.read_excel("pop_age_income_2017.xlsx").drop(
    columns=["INTPTLAT10", "INTPTLON10", "ALAND10"]
)

merge_head = pd.merge(merge_13_14, df_2015, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

merge_head = pd.merge(merge_head, df_2016, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

df_2018 = pd.read_excel("pop_age_income_2018.xlsx").drop(
    columns=["INTPTLAT10", "INTPTLON10", "ALAND10"]
)

merge_head = pd.merge(merge_head, df_2017, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

merge_head = pd.merge(merge_head, df_2018, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

df_2019 = pd.read_excel("pop_age_income_2019.xlsx").drop(
    columns=["INTPTLAT10", "INTPTLON10", "ALAND10"]
)

merge_head = pd.merge(merge_head, df_2019, on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

merge_head_d = merge_head.drop(columns=["INTPTLAT10", "INTPTLON10", "ALAND10"])

df_2020 = pd.read_excel("pop_age_income_test_2020.xlsx")
df_2021 = pd.read_excel("pop_age_income_2021.xlsx").drop(
    columns=["INTPTLAT20", "INTPTLON20", "ALAND20"]
)

merged_final = pd.merge(
    df_2020,
    merge_head_d,
    left_on=["COUNTYFP20", "TRACTCE20", "BLOCKCE20"],
    right_on=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"],
    how="left",
)

merged_final = pd.merge(
    merged_final, df_2021, on=["COUNTYFP20", "TRACTCE20", "BLOCKCE20"]
)

merged_final["STATEFP20"] = 36

merged_final = merged_final.drop(columns=["COUNTYFP10", "TRACTCE10", "BLOCKCE10"])

desired_columns = [
    "STATEFP20",
    "COUNTYFP20",
    "TRACTCE20",
    "BLOCKCE20",
    "GEOID20",
    "INTPTLAT20",
    "INTPTLON20",
    "ALAND20",
    "pop_2013",
    "mean_age_2013",
    "mean_income_2013",
    "pop_2014",
    "mean_age_2014",
    "mean_income_2014",
    "pop_2015",
    "mean_age_2015",
    "mean_income_2015",
    "pop_2016",
    "mean_age_2016",
    "mean_income_2016",
    "pop_2017",
    "mean_age_2017",
    "mean_income_2017",
    "pop_2018",
    "mean_age_2018",
    "mean_income_2018",
    "pop_2019",
    "mean_age_2019",
    "mean_income_2019",
    "pop_2020",
    "mean_age_2020",
    "mean_income_2020",
    "pop_2021",
    "mean_age_2021",
    "mean_income_2021",
]

merged_final = merged_final[desired_columns]

merged_final.to_excel("final_merged.xlsx", index=False)
