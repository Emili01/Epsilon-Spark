package com.epsilonClassifier.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

class DataLoader(spark: SparkSession) {
  import spark.implicits._

  def loadData(trainPath: String, testPath: String, task: String): (DataFrame, DataFrame) = {
    println(s"Loading data from: $trainPath and $testPath")
    val trainRaw = spark.read.parquet(trainPath)
    val testRaw = spark.read.parquet(testPath)

    val (trainProcessed, testProcessed) = task.toLowerCase match {
      case "epsilon" =>
        prepareForEpsilon(trainRaw, testRaw)
      case "category" =>
        prepareForCategory(trainRaw, testRaw) // Tu código original (omitido aquí por brevedad)
      case "sentiment" =>
        prepareForSentiment(trainRaw, testRaw) // Tu código original
      case _ => throw new IllegalArgumentException(s"Unknown task: $task")
    }

    (trainProcessed, testProcessed)
  }

  private def prepareForEpsilon(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
    println("Preparing Epsilon data (Binary Classification)...")
    
    // Tu script Python generó columnas "0", "1", ... "2000"
    // "0" es el label. "1" a "2000" son features.
    
    // 1. Identificar columnas de features (1 a 2000)
    // Nos aseguramos de ordenarlas numéricamente, no alfabéticamente (1, 10, 100...)
    val featureCols = (1 to 2000).map(_.toString).toArray

    // 2. Assembler
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
      // handleInvalid = "keep" o "skip" si hay nulos, aunque Epsilon no debería tenerlos.
    
    def process(df: DataFrame): DataFrame = {
        // Casting inicial: Pandas a veces infiere enteros o strings, aseguramos Double
        var tempDf = df
        
        // Convertir label: Epsilon tiene -1 y 1. Spark necesita 0 y 1.
        // Asumimos que la columna "0" es el label.
        tempDf = tempDf.withColumn("labelRaw", col("0").cast(DoubleType))
                       .withColumn("label", when($"labelRaw" <= 0, 0.0).otherwise(1.0))
        
        // Castear features a Double si no lo son
        featureCols.foreach { c => 
            tempDf = tempDf.withColumn(c, col(c).cast(DoubleType))
        }

        val assembled = assembler.transform(tempDf).select("features", "label")
        assembled
    }

    val trainFinal = process(trainDf)
    val testFinal = process(testDf)

    (trainFinal, testFinal)
  }

  // ... Mantén tus métodos prepareForCategory y prepareForSentiment aquí ...
  private def prepareForCategory(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
      // (Tu código original intacto)
      (trainDf, testDf) 
  }
   private def prepareForSentiment(trainDf: DataFrame, testDf: DataFrame): (DataFrame, DataFrame) = {
      // (Tu código original intacto)
      (trainDf, testDf)
  }
}