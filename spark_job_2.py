from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, min, max, unix_timestamp, to_timestamp, round

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("VehicleLocationMetrics2") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
    .getOrCreate()

# Constants
S3_BUCKET_NAME = "car-rental-marketplace"
AWS_REGION = "eu-west-1"
DATA_FOLDER = "raw-data/"

# Defining S3 folder paths
users_path = f"s3a://{S3_BUCKET_NAME}/{DATA_FOLDER}/users.csv"
locations_path = f"s3a://{S3_BUCKET_NAME}/{DATA_FOLDER}/locations.csv"
vehicles_path = f"s3a://{S3_BUCKET_NAME}/{DATA_FOLDER}/vehicles.csv"
rental_transactions_path = f"s3a://{S3_BUCKET_NAME}/{DATA_FOLDER}/rental_transactions.csv"

# Loading the data from S3
rental_transactions_df = spark.read.option("header", True).csv(rental_transactions_path)
vehicles_df = spark.read.option("header", True).csv(vehicles_path)
users_df = spark.read.option("header", True).csv(users_path)
locations_df = spark.read.option("header", True).csv(locations_path)

# Total transactions per day
daily_transactions = rental_transactions_df.groupBy('rental_start_time', 'user_id').agg(
    round(sum('total_amount'), 2).alias('total_revenue'),
    count('*').alias('total_transactions')
)

# Casting the 'total_transactions' column to int since it was 'long'
daily_transactions = daily_transactions.withColumn(
    "total_transactions", daily_transactions["total_transactions"].cast("int")
)

print("Working for daily_transactions")
print(daily_transactions.show(5))

# Converting rental_start_time and rental_end_time to date format
rental_transactions_df = rental_transactions_df \
    .withColumn("rental_start_time", to_timestamp("rental_start_time")) \
    .withColumn("rental_end_time", to_timestamp("rental_end_time"))

# Filtering out invalid dates (null or invalid format) due to multiple errors in converting to parquet format
rental_transactions_df = rental_transactions_df \
    .filter(col("rental_start_time").isNotNull() & col("rental_end_time").isNotNull())

# Converting the rental duration to hours
rental_transactions_df = rental_transactions_df.withColumn(
    "rental_hours",
    round(((unix_timestamp("rental_end_time") - unix_timestamp("rental_start_time")) / 3600),2).cast("double")
)

# User spending and rental duration
user_metrics = rental_transactions_df.groupBy('user_id').agg(
    round(avg('total_amount'), 2).alias('avg_transaction_value'),
    count('*').alias('total_transactions'),
    round(sum('total_amount'), 2).alias('total_revenue'),
    round(max('total_amount'), 2).alias('max_spending'),
    round(min('total_amount'), 2).alias('min_spending'),
    sum('rental_hours').alias('total_rental_hours')
)

# Casting the 'total_transactions' column to int since it was 'long'
user_metrics = user_metrics.withColumn(
    "total_transactions", user_metrics["total_transactions"].cast("int")
)

# Casting max_spending and min_spending to double
user_metrics = user_metrics.withColumn("max_spending", user_metrics["max_spending"].cast("double"))
user_metrics = user_metrics.withColumn("min_spending", user_metrics["min_spending"].cast("double"))

print("Working for user_metrics")
print(user_metrics.show(5))

# Saving results to S3 in Parquet format
daily_transactions.write.mode("overwrite").parquet("s3a://car-rental-marketplace/processed/daily_transactions/")
user_metrics.write.mode("overwrite").parquet("s3a://car-rental-marketplace/processed/user_metrics/")