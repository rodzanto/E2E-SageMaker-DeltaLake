#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os
import warnings

import pandas as pd
import numpy as np

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/input/code/my_package/requirements.txt",
])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.exceptions import DataConversionWarning
import delta_sharing

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "black",
    "lstat",
    "medv",
]
class_labels = [" - 50000.", " 50000+."]


def print_shape(df):
    negative_examples, positive_examples = np.bincount(df["medv"])
    print(
        "Data shape: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    parser.add_argument("--train", type=str)
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    #input_data_path = os.path.join("/opt/ml/processing/input", "census-income.csv")

    # Take the profile file, create a SharingClient, and read data from the delta lake table
    profile_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(profile_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    profile_file = profile_files[0]
    print(f'Found profile file: {profile_file}')

    # Create a SharingClient
    client = delta_sharing.SharingClient(profile_file)
    table_url = profile_file + "#delta_sharing.default.boston-housing"

    # Load the table as a Pandas DataFrame
    print('Loading boston-housing table from Delta Lake')
    df = delta_sharing.load_as_pandas(table_url)
    print(f'Train data shape: {train_data.shape}')

    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)

    negative_examples, positive_examples = np.bincount(df["medv"])
    print(
        "Data after cleaning: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )

    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("medv", axis=1), df["medv"], test_size=split_ratio, random_state=0
    )

    preprocess = make_column_transformer(
        (
            ["age", "indus"],
            KBinsDiscretizer(encode="onehot-dense", n_bins=10),
        ),
        (["tax"], StandardScaler()),
        (["chas", "rad"], OneHotEncoder(sparse=False)),
    )
    print("Running preprocessing and feature engineering transformations")
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
