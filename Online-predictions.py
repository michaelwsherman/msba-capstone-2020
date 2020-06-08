inputs = {'consumer_complaint_narrative': "In XX/XX/XXXX, I pulled my credit and saw Smith Rouchon & Associates ( XXXX, MS ) had placed a medical collection on my credit, as of XX/XX/XXXX. From XX/XX/XXXX - XX/XX/XXXX I was never contacted to collect this debt or aware it was bought by them, as I'd JUST made a payment to the OC the day before. I've disputed the collection due to the incorrect amount they are trying to collect more than 6 times to no avail of having it removed from my XXXX  credit file. It has been removed from XXXX  and XXXX. As of today, I still have yet to hear from the CA regarding collection and I won't contact them directly as that would reset the collection date for said debt."
           , 'issue': 'False statements or representation'
		   , 'product': 'Debt collection'
		   , 'state': 'MS'
		   , 'subissue': 'Attempted to collect wrong amount'
		   , 'submitted_via': 'Web'
		   , 'subproduct': 'Medical debt'
		   , 'zip_code': '39158'}

# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Makes an online prediction using the AutoML Tables model specified in the config.
Requires the model to be pre-deployed. See the instructions files on details
about deploying the model via the GCP console.
Assumes that a model with the specified display name has already been trained,
and requires the user to provide a new data row for the online prediction. The final 
Prediction is instantaneous.
See https://cloud.google.com/automl-tables/docs/predict for details.
"""

import logging
import sys

import google.api_core.exceptions
from google.cloud import automl_v1beta1 as automl

import utils

logging.basicConfig(level=logging.DEBUG)

def main():
  """
  Executes online prediction using a model trained on AutoML Forecasting.
  Uses parameters specified in the online parameters file, to determine the
  new feature values and predictions.
  Requires the model to be pre-deployed. See the instructions files on details
  about deploying the model via the GCP console.
  See the configuration file for more details.
  1. Runs online prediction operation on AutoML Forecasting service.
  2. Displays the final prediction for 
  """

  config_path = utils.parse_arguments(sys.argv).config_path
  config = utils.read_config(config_path)

  # Defining subconfigs explicitly for readability.
  global_config = config['global']

    
  # Initialize the clients.
  tables_client = automl.TablesClient(project=global_config['destination_project_id'], region=global_config['automl_compute_region'])

  client = automl.TablesClient(project=global_config['destination_project_id'], region=global_config['automl_compute_region'])
  response = client.predict(model_display_name=global_config['model_display_name'], inputs=inputs)

  print("Prediction results:")
  for result in response.payload:
      print(
          "Predicted class name: {}".format(result.tables.value.string_value)
      )
      print("Predicted class score: {}".format(result.tables.score))

if __name__ == '__main__':
  main()