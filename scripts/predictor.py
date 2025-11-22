from typing import Union, List
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def test_svm(
    train_df: Union[pd.DataFrame, List[pd.DataFrame]],
    test_df: pd.DataFrame,
):
    """
    Train an SVM on Adult income dataset and print accuracy on test_df.

    Accuracy = (# correct predictions) / (total test rows)

    train_df can be:
      - one DataFrame, or
      - a list of DataFrames (will be concatenated first)
    """

    # ---------- 1. Merge train DataFrames if list ----------
    if isinstance(train_df, list):
        train_df = pd.concat(train_df, ignore_index=True)

    # ---------- 2. Clean dataset ----------
    def clean(df):
        df = df.copy()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df[(df != "?").all(axis=1)]
        return df

    train_df = clean(train_df)
    test_df  = clean(test_df)

    # ---------- 3. Split X / y ----------
    X_train = train_df.drop(columns=["income"])
    y_train = (train_df["income"].astype(str).str.contains(">50K")).astype(int)

    X_test  = test_df.drop(columns=["income"])
    y_test  = (test_df["income"].astype(str).str.contains(">50K")).astype(int)

    # ---------- 4. Feature types ----------
    numeric_features = [
        "age","fnlwgt","education-num",
        "capital-gain","capital-loss","hours-per-week",
    ]

    categorical_features = [
        "workclass","education","marital-status","occupation",
        "relationship","race","sex","native-country",
    ]

    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ---------- 5. Build SVM pipeline ----------
    svm_clf = Pipeline([
        ("preprocess", preprocessor),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    # ---------- 6. Train ----------
    svm_clf.fit(X_train, y_train)

    # ---------- 7. Predict ----------
    y_pred = svm_clf.predict(X_test)

    # ---------- 8. Accuracy ----------
    accuracy = accuracy_score(y_test, y_pred)
    correct = (y_pred == y_test).sum()
    total = len(y_test)

    print(f"Accuracy: {accuracy:.4f}  ({correct}/{total}) rows correct")

    return svm_clf, accuracy, correct, total


def test_decision_tree(
    train_df: Union[pd.DataFrame, List[pd.DataFrame]],
    test_df: pd.DataFrame,
):
    """
    Train a Decision Tree on Adult income dataset and print accuracy on test_df.

    Accuracy = (# correct predictions) / (total test rows)

    train_df can be:
      - one DataFrame, or
      - a list of DataFrames (will be concatenated first)
    """

    # ---------- 1. Merge train DataFrames if list ----------
    if isinstance(train_df, list):
        train_df = pd.concat(train_df, ignore_index=True)

    # ---------- 2. Clean dataset ----------
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Strip whitespace from all string cells
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # Drop rows with any "?"
        df = df[(df != "?").all(axis=1)]
        return df

    train_df = clean(train_df)
    test_df  = clean(test_df)

    # ---------- 3. Split X / y ----------
    if "income" not in train_df.columns or "income" not in test_df.columns:
        raise ValueError("Both train_df and test_df must contain an 'income' column.")

    X_train = train_df.drop(columns=["income"])
    y_train = (train_df["income"].astype(str).str.contains(">50K")).astype(int)

    X_test  = test_df.drop(columns=["income"])
    y_test  = (test_df["income"].astype(str).str.contains(">50K")).astype(int)

    # ---------- 4. Feature types ----------
    numeric_features = [
        "age", "fnlwgt", "education-num",
        "capital-gain", "capital-loss", "hours-per-week",
    ]

    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    if not numeric_features and not categorical_features:
        raise ValueError("No known numeric or categorical features found in train_df.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ---------- 5. Build Decision Tree pipeline ----------
    tree_clf = Pipeline([
        ("preprocess", preprocessor),
        ("tree", DecisionTreeClassifier(
            random_state=42,
            max_depth=None,          # you can set e.g. 10 to limit depth
            min_samples_leaf=1,
        )),
    ])

    # ---------- 6. Train ----------
    tree_clf.fit(X_train, y_train)

    # ---------- 7. Predict ----------
    y_pred = tree_clf.predict(X_test)

    # ---------- 8. Accuracy ----------
    accuracy = accuracy_score(y_test, y_pred)
    correct = int((y_pred == y_test).sum())
    total = int(len(y_test))

    print(f"Decision Tree Accuracy: {accuracy:.4f}  ({correct}/{total}) rows correct")

    return tree_clf, accuracy, correct, total
