import pandas as pd
import numpy as np

import sklearn.model_selection
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from load_data import load_data


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",
    "Internships",
    "Projects",
    "Workshops",
    "SoftSkillsRating",
    "MockInterviewScore"
]


# ============================================================
# NOMINAL CATEGORICAL COLUMNS
# ============================================================

NOMINAL_COLUMNS = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "ExtraCurricular"
]


# ============================================================
# ORDINAL CATEGORICAL COLUMNS
# ============================================================

ORDINAL_COLUMNS = [
    "CollegeTier",
    "CGPA_Tier"
]


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def run_preprocessing():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    original_shape = df.shape

    # Keep only columns that actually exist in dataset
    numeric_cols = [
        col for col in NUMERIC_COLUMNS
        if col in df.columns
    ]

    nominal_cols = [
        col for col in NOMINAL_COLUMNS
        if col in df.columns
    ]

    ordinal_cols = [
        col for col in ORDINAL_COLUMNS
        if col in df.columns
    ]

    # --------------------------------------------------------
    # 2. REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    if "PlacementStatus" in df.columns:

        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df["PlacementStatus"]
        )

    else:

        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42
        )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # --------------------------------------------------------
    # 4. STANDARD SCALING
    # --------------------------------------------------------

    standard_scaler = StandardScaler()

    if numeric_cols:

        train_standard = standard_scaler.fit_transform(
            train_df[numeric_cols]
        )

        test_standard = standard_scaler.transform(
            test_df[numeric_cols]
        )

        train_standard_df = pd.DataFrame(
            train_standard,
            columns=numeric_cols,
            index=train_df.index
        )

        test_standard_df = pd.DataFrame(
            test_standard,
            columns=numeric_cols,
            index=test_df.index
        )

    else:

        train_standard_df = pd.DataFrame()
        test_standard_df = pd.DataFrame()

    # --------------------------------------------------------
    # 5. MIN-MAX SCALING
    # --------------------------------------------------------

    minmax_scaler = MinMaxScaler()

    if numeric_cols:

        train_minmax = minmax_scaler.fit_transform(
            train_df[numeric_cols]
        )

        test_minmax = minmax_scaler.transform(
            test_df[numeric_cols]
        )

        train_minmax_df = pd.DataFrame(
            train_minmax,
            columns=numeric_cols,
            index=train_df.index
        )

        test_minmax_df = pd.DataFrame(
            test_minmax,
            columns=numeric_cols,
            index=test_df.index
        )

    else:

        train_minmax_df = pd.DataFrame()
        test_minmax_df = pd.DataFrame()

    # --------------------------------------------------------
    # 6. ONE-HOT ENCODING
    # --------------------------------------------------------

    train_ohe_df = pd.DataFrame()
    test_ohe_df = pd.DataFrame()

    if nominal_cols:

        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        train_ohe = ohe.fit_transform(
            train_df[nominal_cols]
        )

        test_ohe = ohe.transform(
            test_df[nominal_cols]
        )

        ohe_columns = ohe.get_feature_names_out(
            nominal_cols
        )

        train_ohe_df = pd.DataFrame(
            train_ohe,
            columns=ohe_columns,
            index=train_df.index
        )

        test_ohe_df = pd.DataFrame(
            test_ohe,
            columns=ohe_columns,
            index=test_df.index
        )

    else:

        ohe_columns = []

    # --------------------------------------------------------
    # 7. ORDINAL ENCODING
    # --------------------------------------------------------

    ordinal_result_train = pd.DataFrame(
        index=train_df.index
    )

    ordinal_result_test = pd.DataFrame(
        index=test_df.index
    )

    ordinal_categories = []

    if "CollegeTier" in ordinal_cols:

        ordinal_categories.append(
            ["Tier3", "Tier2", "Tier1"]
        )

    if "CGPA_Tier" in ordinal_cols:

        ordinal_categories.append(
            ["Low", "Medium", "High"]
        )

    if ordinal_cols:

        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        train_ordinal = ordinal_encoder.fit_transform(
            train_df[ordinal_cols]
        )

        test_ordinal = ordinal_encoder.transform(
            test_df[ordinal_cols]
        )

        ordinal_encoded_columns = [
            col + "_enc"
            for col in ordinal_cols
        ]

        ordinal_result_train = pd.DataFrame(
            train_ordinal,
            columns=ordinal_encoded_columns,
            index=train_df.index
        )

        ordinal_result_test = pd.DataFrame(
            test_ordinal,
            columns=ordinal_encoded_columns,
            index=test_df.index
        )

    else:

        ordinal_encoded_columns = []

    # --------------------------------------------------------
    # 8. OUTLIER DETECTION USING IQR
    # --------------------------------------------------------

    outlier_results = {}

    for column in numeric_cols:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_fence) |
            (df[column] > upper_fence)
        ]

        outlier_results[column] = {
            "q1": round(float(Q1), 3),
            "q3": round(float(Q3), 3),
            "iqr": round(float(IQR), 3),
            "lower_fence": round(float(lower_fence), 3),
            "upper_fence": round(float(upper_fence), 3),
            "outlier_count": int(len(outliers))
        }

    # --------------------------------------------------------
    # 9. OUTLIER CLIPPING
    # --------------------------------------------------------

    clipped_df = df.copy()

    clipping_results = {}

    for column in numeric_cols:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        clipped_df[column] = df[column].clip(
            lower=lower_fence,
            upper=upper_fence
        )

        clipping_results[column] = {
            "before_min": round(float(df[column].min()), 3),
            "after_min": round(float(clipped_df[column].min()), 3),
            "before_max": round(float(df[column].max()), 3),
            "after_max": round(float(clipped_df[column].max()), 3)
        }

    # --------------------------------------------------------
    # 10. SAMPLE DATA FOR DISPLAY
    # --------------------------------------------------------

    standard_preview = (
        train_standard_df
        .head(5)
        .round(3)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    minmax_preview = (
        train_minmax_df
        .head(5)
        .round(3)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    ohe_preview = (
        train_ohe_df
        .head(5)
        .round(3)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    ordinal_preview = (
        ordinal_result_train
        .head(5)
        .round(3)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    # --------------------------------------------------------
    # 11. RETURN RESULTS
    # --------------------------------------------------------

    return {

        "original_rows": int(original_shape[0]),
        "original_columns": int(original_shape[1]),

        "rows_after_duplicates": int(df.shape[0]),

        "duplicate_count": duplicate_count,

        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),

        "train_percentage": 70,
        "test_percentage": 30,

        "numeric_columns": numeric_cols,

        "nominal_columns": nominal_cols,

        "ordinal_columns": ordinal_cols,

        "standard_preview": standard_preview,

        "minmax_preview": minmax_preview,

        "ohe_preview": ohe_preview,

        "ordinal_preview": ordinal_preview,

        "encoded_column_count": len(ohe_columns),

        "encoded_columns": list(ohe_columns),

        "ordinal_encoded_columns": ordinal_encoded_columns,

        "outliers": outlier_results,

        "clipping": clipping_results
    }


if __name__ == "__main__":

    results = run_preprocessing()

    print("\n======================================")
    print("PREPROCESSING RESULTS")
    print("======================================")

    print("Original shape:",
          results["original_rows"],
          "x",
          results["original_columns"])

    print("Duplicate rows:",
          results["duplicate_count"])

    print("Training rows:",
          results["train_rows"])

    print("Testing rows:",
          results["test_rows"])

    print("\nStandard Scaling:")
    print(pd.DataFrame(results["standard_preview"]))

    print("\nMin-Max Scaling:")
    print(pd.DataFrame(results["minmax_preview"]))

    print("\nOne-Hot Encoded columns:")
    print(results["encoded_columns"])

    print("\nOrdinal Encoding:")
    print(pd.DataFrame(results["ordinal_preview"]))

    print("\nOutlier Results:")
    for column, result in results["outliers"].items():
        print(column, result)