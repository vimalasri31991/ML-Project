import os
import pandas as pd

# df = pd.read_csv(r"C:\Users\HP\PycharmProjects\MLProject\placement_predict_50k Dataset (3)(in).csv")
# # print(df)
# # print(df.dtypeZZs)
# print(df[['CGPA','Projects','Internships']].describe())
# #Display the first 5 rows
# print(df.head())
#
# #Display the shape (rows, columns)
# print(df.shape)
#
# print(df['Projects'].tolist())


DATA_PATH = r"C:\Users\HP\PycharmProjects\MLProject\placement_predict_50k Dataset (3)(in).csv"


def load_data(path: str=DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def get_data_summary(path: str=DATA_PATH) -> dict:
    df = load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns" : list(df.columns),
        "dtypes" : {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts":{col: int(df[col].isnull().sum()) for col in df.columns},
        "preview" : df.head(10).to_dict(orient="records"),
    }
    return summary

if __name__ == '__main__':
    data = load_data()
    print(get_data_summary())