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

# Import generic functions...
import argparse
import os
import warnings

import pandas as pd
import numpy as np

import sys
import subprocess

# Install and import dependencies...
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/input/code/my_package/requirements.txt",
])
import delta_sharing

# Import SKLearn libraries...
from sklearn.compose import make_column_transformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-file", type=str)
    parser.add_argument("--table", type=str)
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    # Take the Delta Sharing profile file and create a SharingClient...
    profile_files = [os.path.join(args.profile_file, file) for file in os.listdir(args.profile_file)]
    if len(profile_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.profile_file, "profile-file")
        )
    profile_file = profile_files[0]
    print(f'Found profile file: {profile_file}')
    client = delta_sharing.SharingClient(profile_file)
    table_url = profile_file + args.table

    # Load the Delta Table as a Pandas DataFrame...
    print(f'Loading {args.table} table from Delta Lake')
    df = delta_sharing.load_as_pandas(table_url)
    print(f'Train data shape: {df.shape}')

    # Perform some sample transformations - Replace here with your own transformations...
    # ---
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    negative_examples, positive_examples = np.bincount(df["chas"])
    print(
        "Data after cleaning: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )

    processed_features = df

    # ---
    # Write processed data after transformations...
    processed_features_output_path = os.path.join("/opt/ml/processing/processed", "processed_features.csv")

    print("Saving processed features to {}".format(processed_features_output_path))
    pd.DataFrame(processed_features).to_csv(processed_features_output_path, header=True, index=False)
