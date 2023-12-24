-- replace project id with your project id
-- replace dataset id with the dataset id that you want to create. 
-- To know more about these parameters - refer to https://cloud.google.com/bigquery/docs/datasets#sql

CREATE SCHEMA PROJECT_ID.DATASET_ID
  OPTIONS (
    default_kms_key_name = 'KMS_KEY_NAME',
    default_partition_expiration_days = PARTITION_EXPIRATION,
    default_table_expiration_days = TABLE_EXPIRATION,
    description = 'DESCRIPTION',
    labels = [('LABEL_1','VALUE_1'),('LABEL_2','VALUE_2')],
    location = 'LOCATION',
    max_time_travel_hours = HOURS,
    storage_billing_model = BILLING_MODEL);