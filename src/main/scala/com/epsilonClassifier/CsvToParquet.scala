package com.epsilonClassifier

import org.apache.spark.sql.SparkSession

object CsvToParquet {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("CsvToParquet")
      .master("local[*]")
      .config("spark.driver.memory", "8g")
      .getOrCreate()

    println("Converting data/train.csv to data/train.parquet...")
    val trainDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/train.csv")
    trainDf.write.mode("overwrite").parquet("data/train.parquet")
    println("Conversion of training data complete.")

    println("Converting data/test.csv to data/test.parquet...")
    val testDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/test.csv")
    testDf.write.mode("overwrite").parquet("data/test.parquet")
    println("Conversion of test data complete.")

    spark.stop()
    println("Parquet conversion finished.")
  }
}
