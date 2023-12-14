import logging
import pickle
from math import log, radians

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmr10"


def process_hospital_data(df):
    """
    Process hospital data.
    """
    df["Facility Open Date"] = pd.to_datetime(df["Facility Open Date"])
    df = df.sort_values(by="Facility Open Date")

    df = df[
        df["Facility City"].isin(
            ["New York", "Brooklyn", "Bronx", "Staten Island", "Queens"]
        )
    ].reset_index(drop=True)

    return df


def process_blocks_data(blocks):
    """
    Process blocks data.
    """
    blocks = blocks.drop(
        columns=[
            "STATEFP20_y",
            "COUNTYFP20_y",
            "TRACTCE20_y",
            "BLOCKCE20_y",
            "INTPTLAT20_y",
            "INTPTLON20_y",
        ]
    )

    # rename columns
    blocks = blocks.rename(
        columns={
            "STATEFP20_x": "STATEFP20",
            "COUNTYFP20_x": "COUNTYFP20",
            "TRACTCE20_x": "TRACTCE20",
            "BLOCKCE20_x": "BLOCKCE20",
            "INTPTLAT20_x": "INTPTLAT20",
            "INTPTLON20_x": "INTPTLON20",
        }
    )

    # convert m2 to km2
    blocks["ALAND20"] = blocks["ALAND20"] / 10**6

    blocks_updated = blocks[
        blocks[[col for col in blocks.columns if "pop" in col]]
        .isna()
        .sum(axis="columns")
        == 0
    ]

    blocks_updated = blocks_updated[
        (blocks_updated["ALAND20"].between(0, 0.1, inclusive="neither"))
    ].copy()

    return blocks_updated


def get_nearest_col(block, col, year):
    """
    Get the nearest column value for a given year.
    """
    while True:
        try:
            return block[col + "_" + str(year)]

        except KeyError:
            year -= 1
            if year < 2000:
                raise KeyError(f"Invalid Year {year} for {col}")


def get_population_r_squared(block, population_threshold, date):
    """
    Get the population radius squared.
    """
    return (
        population_threshold
        * block["ALAND20"]
        / get_nearest_col(block, "pop", date.year)
    )


def get_population_score(block, distance_threshold, date):
    """
    Get the population score.
    """
    population_radius_squared = block["population_radius_squared"]
    distance_min_squared = min(population_radius_squared, distance_threshold**2)

    return 1 / distance_min_squared


def get_age_score(block, age_function, date):
    """
    Get the age score.
    """
    hospital_indicator = age_function(get_nearest_col(block, "mean_age", date.year))
    return hospital_indicator


def get_income_score(block, income_function, date):
    """
    Get the income score.
    """
    return income_function(get_nearest_col(block, "mean_income", date.year))


def get_covid_hosprate_score(block, covid_hosprate_func, date):
    """
    Get the covid hospital rate score.
    """
    hospital_indicator = covid_hosprate_func(
        get_nearest_col(block, "covid_hosprate", date.year)
    )
    return hospital_indicator


def get_cardio_score(block, cardio_func, date):
    """
    Get the cardio score.
    """
    return cardio_func(get_nearest_col(block, "cardio", date.year))


def compute_haversine_distance(x, y):
    """
    Compute the haversine distance between two points.
    """
    x_rad = [radians(_) for _ in x]
    y_rad = [radians(_) for _ in y]
    return haversine_distances([x_rad, y_rad])[0][1] * 6371000 / 1000


def loss_function_decay(y_true, y_pred, time_delta):
    """
    Compute the loss function with a time decay.
    """
    distances = []

    for _, block in y_pred.iterrows():
        block_location = (block["Facility Latitude"], block["Facility Longitude"])
        nearest_hospital_distance = float("inf")
        y_true_valid = y_true[
            (y_true["Facility Open Date"] >= block["Facility Open Date"] - time_delta)
            & (y_true["Facility Open Date"] <= block["Facility Open Date"] + time_delta)
        ]

        for _, hospital in y_true_valid.iterrows():
            hospital_location = (
                hospital["Facility Latitude"],
                hospital["Facility Longitude"],
            )
            distance = compute_haversine_distance(block_location, hospital_location)

            if distance < nearest_hospital_distance:
                nearest_hospital_distance = distance
        distances.append(nearest_hospital_distance)

    mean_distance = sum(distances) / len(distances)
    return mean_distance


def loss_function_no_time_constraint(y_true, y_pred, time_delta):
    """
    Compute the loss function without a time decay.
    """
    block_locations = np.column_stack(
        (y_pred["Facility Latitude"], y_pred["Facility Longitude"])
    )
    hospital_locations = np.column_stack(
        (y_true["Facility Latitude"], y_true["Facility Longitude"])
    )

    distances = (
        haversine_distances(np.radians(block_locations), np.radians(hospital_locations))
        * 6371000
        / 1000
    )

    nearest_hospital_distances = np.min(distances, axis=1)

    mean_distance = np.mean(nearest_hospital_distances)
    return mean_distance


def loss_function_corresponding(y_true, y_pred, time_delta):
    """
    Compute the loss function with a time decay.
    """
    num_y_true = len(y_true)
    num_y_pred = len(y_pred)

    if num_y_true < num_y_pred:
        y_pred = y_pred[-num_y_true:]
    elif num_y_true > num_y_pred:
        y_true = y_true[-num_y_pred:]

    block_locations = np.column_stack(
        (y_pred["Facility Latitude"], y_pred["Facility Longitude"])
    )
    hospital_locations = np.column_stack(
        (y_true["Facility Latitude"], y_true["Facility Longitude"])
    )

    distances = (
        haversine_distances(np.radians(block_locations), np.radians(hospital_locations))
        * 6371000
        / 1000
    )

    nearest_hospital_distances = [distances[i][i] for i in range(len(distances))]

    mean_distance = np.mean(nearest_hospital_distances)
    return mean_distance


def penalize_proximity_to_hospitals(block, hospital_locations, distance_threshold):
    """
    Penalizes a block for being too close to a hospital.
    """
    block_location = np.array([block["INTPTLAT20"], block["INTPTLON20"]])
    hospital_locations = np.array(
        hospital_locations[["Facility Latitude", "Facility Longitude"]]
    )

    distances = [
        (
            compute_haversine_distance(block_location, hospital) ** 2
            <= distance_threshold
        )
        for hospital in hospital_locations
    ]

    num_rows_less_than_threshold = np.sum(distances, axis=0)

    return num_rows_less_than_threshold


def penalize_proximity_to_hospitals_ANN(block, nn_model):
    """
    Penalizes a block for being too close to a hospital with approximate nearest neighbors.
    """
    query_block_location = np.array([block["INTPTLAT20"], block["INTPTLON20"]])
    distance_threshold = block["population_radius_squared"]

    return len(
        nn_model.radius_neighbors(
            [query_block_location],
            radius=distance_threshold**0.5,
            return_distance=False,
        )[0]
    )


def build_hospital(
    all_blocks,
    hospital_locations,
    distance_threshold,
    population_threshold,
    age_function,
    income_function,
    cardio_func,
    covid_hosprate_func,
    coefficients,
    CURRENT_YEAR,
):
    """
    Build a hospita

    """
    blocks = all_blocks.copy()

    hospital_locations = np.array(
        hospital_locations[["Facility Latitude", "Facility Longitude"]]
    )

    nn_model = NearestNeighbors(
        n_neighbors=len(hospital_locations),
        radius=distance_threshold**0.5,
        algorithm="ball_tree",
        n_jobs=6,
    ).fit(hospital_locations)

    blocks["population_radius_squared"] = blocks.apply(
        lambda x: get_population_r_squared(x, population_threshold, CURRENT_YEAR),
        axis=1,
    )

    blocks["population_score"] = blocks.apply(
        lambda x: get_population_score(x, distance_threshold, CURRENT_YEAR),
        axis=1,
    )

    blocks["penalty"] = blocks.apply(
        lambda x: penalize_proximity_to_hospitals_ANN(x, nn_model),
        axis=1,
    )

    blocks["age_score"] = blocks.apply(
        lambda x: get_age_score(x, age_function, CURRENT_YEAR), axis=1
    )

    blocks["income_score"] = blocks.apply(
        lambda x: get_income_score(x, income_function, CURRENT_YEAR), axis=1
    )

    blocks["cardio_score"] = blocks.apply(
        lambda x: get_cardio_score(x, cardio_func, CURRENT_YEAR), axis=1
    )

    blocks["covid_hosprate_score"] = blocks.apply(
        lambda x: get_covid_hosprate_score(x, covid_hosprate_func, CURRENT_YEAR), axis=1
    )

    blocks[
        [
            "population_score",
            "penalty",
            "age_score",
            "income_score",
            "cardio_score",
            "covid_hosprate_score",
        ]
    ] = MinMaxScaler().fit_transform(
        blocks[
            [
                "population_score",
                "penalty",
                "age_score",
                "income_score",
                "cardio_score",
                "covid_hosprate_score",
            ]
        ]
    )

    blocks["score"] = np.dot(
        coefficients,
        blocks[
            [
                "population_score",
                "age_score",
                "income_score",
                "cardio_score",
                "covid_hosprate_score",
                "penalty",
            ]
        ]
        .to_numpy()
        .T,
    )

    blocks["score"] = MinMaxScaler().fit_transform(
        PowerTransformer().fit_transform(blocks["score"].to_numpy().reshape(-1, 1))
    )

    selected_block = blocks[blocks["score"] >= 0.8].sample(1)

    hospital_location = selected_block[["INTPTLAT20", "INTPTLON20"]].values[0]

    return hospital_location, selected_block["score"]


def simulate_hospital_building(
    blocks,
    distance_threshold,
    population_threshold,
    age_function,
    income_function,
    cardio_func,
    covid_hosprate_func,
    coefficients,
    loss_function,
    START_YEAR,
    CURRENT_YEAR,
    END_YEAR,
    inter_build_time,
    PROXIMITY_TIME_DELTA,
):
    hospitals = (
        df[
            (df["Facility Open Date"] >= START_YEAR)
            & (df["Facility Open Date"] <= CURRENT_YEAR)
        ]
        .dropna(subset=["Facility Latitude", "Facility Longitude"])
        .copy()
    )
    loss = []
    scores = []

    try:
        while CURRENT_YEAR < END_YEAR:
            CURRENT_YEAR += inter_build_time

            hospital_location, score = build_hospital(
                blocks,
                hospitals,
                distance_threshold,
                population_threshold,
                age_function,
                income_function,
                cardio_func,
                covid_hosprate_func,
                coefficients,
                CURRENT_YEAR,
            )

            hospitals = pd.concat(
                [
                    hospitals,
                    pd.DataFrame(
                        {
                            "Facility Name": "Hospital " + str(len(hospitals) + 1),
                            "Facility Open Date": CURRENT_YEAR,
                            "Facility Latitude": hospital_location[0],
                            "Facility Longitude": hospital_location[1],
                        },
                        index=[0],
                        columns=hospitals.columns,
                    ),
                ],
                ignore_index=True,
            )
            scores.append(score)
            loss.append(
                loss_function(
                    df[
                        (df["Facility Open Date"] >= START_YEAR)
                        & (df["Facility Open Date"] <= CURRENT_YEAR)
                    ].dropna(subset=["Facility Latitude", "Facility Longitude"]),
                    hospitals,
                    PROXIMITY_TIME_DELTA,
                )
            )
    except KeyboardInterrupt:
        return hospitals, loss, scores

    return hospitals, loss, scores


def age_function(x):
    return (-2.31854933 * x) + 0.02567851 * (x**2) + 95.60891791326296


def income_function(income):
    return log(income)


def cardio_function(cases):
    return cases


def covid_hosprate_function(cases):
    return cases


def run_simulation_grid(
    blocks_updated_sample,
    distance_thresholds,
    population_thresholds,
    pop_coeff,
    age_coeff,
    income_coeff,
    cardio_coeff,
    covid_hosprate_coeff,
    penalties,
    loss_functions,
    age_function,
    income_function,
    cardio_function,
    covid_hosprate_function,
    START_YEAR,
    CURRENT_YEAR,
    END_YEAR,
    inter_build_time,
    PROXIMITY_TIME_DELTA,
):
    """
    Run the simulation for all combinations of the given parameters.
    """
    for distance_threshold in distance_thresholds:
        for population_threshold in population_thresholds:
            for pop_c in pop_coeff:
                for age_c in age_coeff:
                    for income_c in income_coeff:
                        for cardio_c in cardio_coeff:
                            for covid_hosprate_c in covid_hosprate_coeff:
                                for penalty in penalties:
                                    for loss_function in tqdm(loss_functions):
                                        coefficients = [
                                            pop_c,
                                            age_c,
                                            income_c,
                                            cardio_c,
                                            covid_hosprate_c,
                                            penalty,
                                        ]
                                        (
                                            generated_hospitals,
                                            loss_values,
                                            scores,
                                        ) = simulate_hospital_building(
                                            blocks_updated_sample,
                                            distance_threshold,
                                            population_threshold,
                                            age_function,
                                            income_function,
                                            cardio_function,
                                            covid_hosprate_function,
                                            coefficients,
                                            loss_function,
                                            START_YEAR,
                                            CURRENT_YEAR,
                                            END_YEAR,
                                            inter_build_time,
                                            PROXIMITY_TIME_DELTA,
                                        )
                                        key = tuple(
                                            [distance_threshold, population_threshold]
                                            + coefficients
                                            + [loss_function.__name__]
                                        )
                                        results[key] = {
                                            "hospitals": generated_hospitals,
                                            "losses": loss_values,
                                            "scores": scores,
                                        }

                                        file_name = "_".join([str(_) for _ in key])

                                        with open(
                                            f"results_simulation_{file_name}.pkl", "wb"
                                        ) as f:
                                            pickle.dump(results, f)
    return results


if __name__ == "__main__":
    # read datasets
    # demographic datasets
    covid = pd.read_excel("../data/COVID_hosprate_by_year_inner.xlsx")
    cardio = pd.read_excel("../data/HeartAttacks_hosprate_by_year_inner.xlsx")
    demo = pd.read_excel("../data/final_merged_9_dec.xlsx")

    # read hospital dataset
    df = pd.read_csv("../../data/Health_Facility_General_Information_20231025.csv")

    # create a single health dataset
    health_data = pd.concat([cardio, covid.iloc[:, 8:]], axis="columns")

    # merge health and demographic datasets
    blocks = pd.merge(demo, health_data, on="GEOID20", how="inner")

    # process datasets
    blocks_updated = process_blocks_data(blocks)
    df = process_hospital_data(df)

    # set simulation parameters
    START_YEAR = pd.to_datetime("2019-01-01")
    CURRENT_YEAR = pd.to_datetime("2020-01-01")
    END_YEAR = pd.to_datetime("2022-01-01")

    DATA_SET_START = df["Facility Open Date"].min()
    DATA_SET_END = START_YEAR

    TOTAL_HOSPITALS = len(df)
    PROXIMITY_TIME_DELTA = pd.Timedelta(days=365 * 2 / 12)
    inter_build_time = pd.Timedelta(
        days=(DATA_SET_END - DATA_SET_START).days / TOTAL_HOSPITALS
    )

    distance_thresholds = [1, 2, 5]
    population_thresholds = [300, 800, 1500]
    pop_coeff = [0.1, 0.33, 0.5]
    age_coeff = [0.4, 0.34, 0.25]
    income_coeff = [0.2, 0.33, 0.25]
    covid_hosprate_coeff = [0.1, 0.33, 0.25]
    cardio_coeff = [0.3, 0.33, 0.25]
    penalties = [-0.8, -0.9, -1.0]

    loss_functions = [
        loss_function_decay,
        loss_function_no_time_constraint,
        loss_function_corresponding,
    ]

    results = {}

    # run simulation
    results = run_simulation_grid(
        blocks_updated,
        distance_thresholds,
        population_thresholds,
        pop_coeff,
        age_coeff,
        income_coeff,
        cardio_coeff,
        covid_hosprate_coeff,
        penalties,
        loss_functions,
        age_function,
        income_function,
        cardio_function,
        covid_hosprate_function,
        START_YEAR,
        CURRENT_YEAR,
        END_YEAR,
        inter_build_time,
        PROXIMITY_TIME_DELTA,
    )
