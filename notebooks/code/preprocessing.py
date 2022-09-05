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

import delta_sharing
from sklearn.compose import make_column_transformer

import boto3
from decimal import Decimal

# Defining some functions for efficiently ingesting into SageMaker Feature Store...
def remove_exponent(d):
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

def transform_row(columns, row) -> list:
    record = []
    for column in columns:
        feature = {'FeatureName': column, 'ValueAsString': str(remove_exponent(Decimal(str(row[column]))))}
        # We can't ingest null value for a feature type into a feature group
        if str(row[column]) not in ['NaN', 'NA', 'None', 'nan', 'none']:
            record.append(feature)
    # Complete with EventTime feature
    timestamp = {'FeatureName': 'EventTime', 'ValueAsString': str(pd.to_datetime("now").timestamp())}
    record.append(timestamp)
    return record

def ingest_to_feature_store(fg, rows) -> None:
    session = boto3.session.Session(region_name='eu-west-1')
    featurestore_runtime_client = session.client(service_name='sagemaker-featurestore-runtime')
    columns = rows.columns
    for index, row in df.iterrows():
        record = transform_row(columns, row)
        #print(f'Putting record:{record}')
        response = featurestore_runtime_client.put_record(FeatureGroupName=fg, Record=record)
        #print(f'Done with row:{index}')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200

# Main...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-file", type=str)
    parser.add_argument("--table", type=str)
    parser.add_argument("--feature-group", type=str)
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    # Take the Delta Sharing profile file and create a SharingClient...
    profile_files = [os.path.join(args.profile_file, file) for file in os.listdir(args.profile_file)]
    if len(profile_files) == 0:
        raise ValueError(
            ("There are no files in {}.\n").format(args.profile_file, "profile-file")
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
    
    # Ingesting the resulting data into our Feature Group...
    print(f"Ingesting processed features into Feature Group {args.feature_group}...")
    ingest_to_feature_store(args.feature_group, processed_features)
    print("All done.")
