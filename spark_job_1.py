from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, sum, avg, min, max, unix_timestamp, to_timestamp, round

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("VehicleLocationMetrics1") \
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

# Revenue per location
revenue_per_location = rental_transactions_df.groupBy("pickup_location").agg(round(sum("total_amount"), 2).alias("total_revenue"))

# Transactions per location
transactions_per_location = rental_transactions_df.groupBy("pickup_location").agg(count("*").alias("total_transactions"))

# Casting the 'total_transactions' column to int since it was 'long'
transactions_per_location = transactions_per_location.withColumn(
    "total_transactions", transactions_per_location["total_transactions"].cast("int")
)

# Min/Max/Average transaction amounts
transaction_stats = rental_transactions_df.groupBy("pickup_location").agg(
    round(avg("total_amount"), 2).alias("avg_transaction"),
    round(min("total_amount"), 2).alias("min_transaction"),
    round(max("total_amount"), 2).alias("max_transaction")
)

# Casting min_transaction and max_transaction to double
transaction_stats = transaction_stats.withColumn("min_transaction", col("min_transaction").cast("double"))
transaction_stats = transaction_stats.withColumn("max_transaction", col("max_transaction").cast("double"))

# Unique vehicles used per location
unique_vehicles = rental_transactions_df.groupBy("pickup_location").agg(countDistinct("vehicle_id").alias("unique_vehicles"))

# Casting the 'unique_vehicles' column to int since it was 'long'
unique_vehicles = unique_vehicles.withColumn(
    "unique_vehicles", unique_vehicles["unique_vehicles"].cast("int")
)

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

rental_duration_metrics = rental_transactions_df.join(vehicles_df, rental_transactions_df.vehicle_id == vehicles_df.vehicle_id) \
    .join(locations_df, rental_transactions_df.pickup_location == locations_df.location_id) \
    .groupBy('pickup_location', 'vehicle_type') \
    .agg(
        sum('total_amount').alias('total_revenue'),
        round(avg('rental_hours'),2).alias('avg_rental_duration')
    )

# Merge all metrics
location_metrics = revenue_per_location \
    .join(transactions_per_location, "pickup_location") \
    .join(transaction_stats, "pickup_location") \
    .join(unique_vehicles, "pickup_location") \
    .orderBy("pickup_location")

# Saving results to s3 in parquet format
location_metrics.write.mode("overwrite").parquet("s3a://car-rental-marketplace/processed/location_metrics/")
rental_duration_metrics.write.mode("overwrite").parquet("s3a://car-rental-marketplace/processed/rental_duration_metrics/")

