import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R,    StructField as Fld, \
                              DoubleType as Dbl,  StringType as Str, \
                              IntegerType as Int, DateType as Date, TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')
print('>> Read Out Config-Infos from [dl.cfg]')

os.environ['AWS_ACCESS_KEY_ID']     = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """HELP for procedure CREATE_SPARK_SESSION

    Creation of or retrieves Spark Session
    
    Parameters: 
    ( none )
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.10.0") \
        .getOrCreate()
    return spark
    # @ EMR (AWS): "emr-5.31.0" > "hadoop 2.10.0"


def process_song_data(spark, input_data, output_data):
    """HELP for procedure PROCESS_SONG_DATA

    Reading and Writing Song Data from and to S3 AWS-Bucket. 
    It does extract the songs and artist tables. 
    At the end, uploaded back to S3.
    
    Parameters: 
      spark: spark session
      input_data: data path of song_data JSON files with songs metadata
      output_data: dimensional tables in PARQUET format saving at S3 AWS-Bucket
    """
    # get filepath to song data file [song_data/A/B/C/TRABCEI128F424C983.json]
    # AWS 
    #song_data = input_data + 'song_data/*/*/*/*.json'
    # LOC
    song_data = input_data + 'song_data_unzipped/song_data/*/*/*/*.json'

    # define the song schema (like [staging_songs] table)
    songSchema = R([
    Fld("artist_id",        Str()),
    Fld("artist_latitude",  Dbl()),
    Fld("artist_location",  Str()),
    Fld("artist_longitude", Dbl()),
    Fld("artist_name",      Str()),
    Fld("duration",         Dbl()),
    Fld("num_songs",        Int()),
    Fld("song_id",          Str()),
    Fld("title",            Str()),
    Fld("year",             Int()),
    ])
    
    # read song data JSON file into data frame
    df = spark.read.json(song_data, schema=songSchema)
    print('>> [' + str(df.count()) + '] songs from song_data read out in JSON-format')

    # extract columns to create songs table
    song_columns = ["song_id", "title", "artist_id", "year", "duration"]
    songs_table = df.select(song_columns) \
                    .dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id') \
                     .parquet(output_data + 'songs_data/songs_table.parquet', 'overwrite')

    # extract columns to create artists table
    artists_column = ["artist_id", "artist_name as name", "artist_location as location", \
                      "artist_latitude as latitude", "artist_longitude as longitude"]
    artists_table = df.selectExpr(artists_column).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists_data/artists_table.parquet', 'overwrite')


def process_log_data(spark, input_data, output_data):
    """HELP for procedure PROCESS_LOG_DATA

    Reading and Writing Log Data from and to S3 AWS-Backet.
    It does extract the users, time and songplays tables.
    At the end, uploaded back to S3.
    It does also use the output of process_songs_data function.
    
    Parameters: 
      spark: spark session
      input_data: data path of log_data JSON files with events data
      output_data: dimensional tables in PARQUET format saving at S3 AWS-Bucket
    """
    # AWS:
    # get filepath to log data file [log_data/2018/11/2018-11-12-events.json]
    #log_data = input_data + 'log_data/*/*/*.json'
    # LOC:
    log_data = input_data + 'log_data_unzipped/*.json'

    # read log data file into data frame
    df = spark.read.json(log_data)
    print('>> [' + str(df.count()) + '] logs entries read IN, of JSON logs_data')
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')
    print('>> [' + str(df.count()) + '] songs filtered page("NextSong") of logs_data')

    # extract columns for USERS table    
    # artists_table =  # TYPO at resources (*.zip)!
    users_columns = ["userId as user_id", "firstName as first_name", \
                     "lastName as last_name", "gender", "level"]
    users_table = df.selectExpr(users_columns).dropDuplicates()
    
    # extract columns for USERS table    
    # !! artists_table =  # TYPO at resources (*.zip) !!
    users_columns = ["userId as user_id", "firstName as first_name", \
                     "lastName as last_name", "gender", "level"]
    users_table = df.selectExpr(users_columns).dropDuplicates()

    # write USERS table to parquet files
    users_table.write.parquet(output_data + 'users_data/users_table.parquet', 'overwrite')

    # create "timestamp" column from original timestamp ("ts") column
    get_datetime = udf(lambda x: str( datetime.fromtimestamp( int(x)/1000.0 )) )
    dfTimestamp = df.withColumn("start_time", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = dfTimestamp.select("start_time").dropDuplicates() \
                            .withColumn("hour",    hour(col("start_time"))) \
                            .withColumn("day",     dayofmonth(col("start_time"))) \
                            .withColumn("week",    weekofyear(col("start_time"))) \
                            .withColumn("month",   month(col("start_time"))) \
                            .withColumn("year",    year(col("start_time"))) \
                            .withColumn("weekday", date_format(col("start_time"), 'E'))
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month") \
              .parquet(output_data + 'time_data/time_table.parquet', 'overwrite')

    # preparations for songplays_table
    get_datetime = udf(lambda x: str( datetime.fromtimestamp( int(x)/1000.0 )) )
    df = df.withColumn("start_time", get_datetime(df.ts))
    df = df.withColumn("year",       year(col("start_time")))
    df = df.withColumn("month",      month(col("start_time")))

    print('>> [' + str(df.count()) + '] songs filtered "NextSong" of logs_data')

    # read in song data to use for songplays table
    song_df = spark.read.option("mergeSchema", "true") \
                   .parquet(output_data + "songs_data/songs_table.parquet")

    print('>> [' + str(song_df.count()) + '] songs readout from songs_table.PARQUET')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.join(df, (song_df.title == df.song) ) \
                            .select('start_time', \
                                    df.year, \
                                    df.month, \
                                    col('userId').alias("user_id"), \
                                    df.level, \
                                    song_df.song_id, \
                                    song_df.artist_id, \
                                    col('sessionId').alias("session_id"), \
                                    df.location, \
                                    col('userAgent').alias("user_agent") \
                                    )
    songplays_table = songplays_table.withColumn("songplay_id", monotonically_increasing_id())    

    print('>> [' + str(songplays_table.count()) + '] songs found, on JOIN matching for songsplays_table')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month") \
                   .parquet(output_data + 'songplays_data/songplays_table.parquet', 'overwrite')


def main():
    """HELP for ETL (main)

    Extract songs and events data from S3, transform it into dimensional tables
    format and uploads back into S3 in PARQUET format
    
    Subprocesses
    - create_spark_session
    - process_song_data
    - process_log_data
    """
    print('>> START (main)')

    spark = create_spark_session()
    print('>> spark session created')

    # AWS Settings @ EMR Notebook Environment
    #input_data = "s3a://udacity-dend/"
    #output_data = "s3a://udacity-data-lake/output/"
    ##output_data = "s3a://data-lake-project-out/"
    # + + + + + 
    # LOC Setting @ Udacity Workbook Environment
    input_data  = "data/"
    output_data = "data_output/"
    
    print('>> processing song data')
    process_song_data(spark, input_data, output_data)
    print('>> song data processed')

    print('>> processing log data')
    process_log_data(spark, input_data, output_data)
    print('>> log data processed')

    print('>> END!')


if __name__ == "__main__":
    main()
