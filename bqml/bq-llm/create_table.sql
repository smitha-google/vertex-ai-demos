-- This is an example of loading a csv file into a BQ table. 
-- Have your data loaded into a GCS Bucket and get the gs URI handy
-- Replace mydataset with the dataset where you wnat to create the table
-- Replace mytable with the table name
-- Refer https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv#loading_csv_data_into_a_table for learning more
LOAD DATA OVERWRITE mydataset.mytable
(x INT64,y STRING)
FROM FILES (
  format = 'CSV',
  uris = ['gs://bucket/path/file.csv']);