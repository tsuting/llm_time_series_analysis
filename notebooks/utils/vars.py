from pathlib import Path

DATA_DIR = "../../data/"
EXCEPT_FILES = [
    "easy_questions.csv",
    "medium_questions.csv",
    "hard_questions.csv",
]

QUESTION_FILES = [
    Path(DATA_DIR, "easy_questions.csv"),
    Path(DATA_DIR, "medium_questions.csv"),
]
