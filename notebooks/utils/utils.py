import pandas as pd
from typing import Any
from sklearn.metrics import accuracy_score


def convert_types(element: any) -> Any:
    """ "
    Convert element to float or datetime if possible
    """
    if element is None:
        return element

    try:
        return float(element)
    except ValueError:
        try:
            return pd.to_datetime(element)
        except:
            return element


def eval(df: pd.DataFrame, details: bool = False) -> pd.DataFrame:
    # loop through each file
    for file in df["file"].unique():
        df_file = df[df["file"] == file]
        print("=" * 50)
        print(
            f"File: {file}; Accuracy: {accuracy_score(df_file['answer_true'].astype(str).tolist(), df_file['answer_pred'].astype(str).tolist())}"
        )
        print("=" * 50)
        # loop through each question with wrong answer
        for _, row in df_file[
            df_file["answer_true"] != df_file["answer_pred"]
        ].iterrows():
            print(f"question: {row['question']}")
            print(f"answer_pred: {row['answer_pred']}")
            print(f"answer_true: {row['answer_true']}")
            if details:
                print(f'messages: {"\n".join(row["messages"])}')
                print(f'steps: {row["steps"]}')
            print("*" * 50)

        del df_file
