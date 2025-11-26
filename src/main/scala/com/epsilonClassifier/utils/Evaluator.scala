package com.epsilonClassifier.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._

case class Metrics(
  accuracy: Double,
  precision: Double,
  recall: Double,
  f1: Double,
  confusionMatrix: Array[Array[Double]],
  predictions: DataFrame
)

class Evaluator(spark: SparkSession, task: String) {
  import spark.implicits._

  def evaluate(model: Transformer, data: DataFrame, split: String): Metrics = {
    // Make predictions
    val predictions = model.transform(data)

    // Calculate accuracy
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = accuracyEvaluator.evaluate(predictions)

    // Calculate weighted precision
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val precision = precisionEvaluator.evaluate(predictions)

    // Calculate weighted recall
    val recallEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val recall = recallEvaluator.evaluate(predictions)

    // Calculate F1 score
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val f1 = f1Evaluator.evaluate(predictions)

    // Get confusion matrix
    val predictionAndLabels = predictions
      .select($"prediction", $"label")
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val confusionMatrix = metrics.confusionMatrix.toArray
      .grouped(metrics.confusionMatrix.numCols)
      .toArray

    println(s"\n=== $split Metrics ===")
    println(f"Accuracy: $accuracy%.4f")
    println(f"Precision: $precision%.4f")
    println(f"Recall: $recall%.4f")
    println(f"F1 Score: $f1%.4f")
    println("\nConfusion Matrix:")
    confusionMatrix.foreach(row => println(row.mkString("\t")))

    Metrics(accuracy, precision, recall, f1, confusionMatrix, predictions)
  }

  def getClassificationReport(metrics: Metrics): String = {
    val numClasses = metrics.confusionMatrix.length
    val report = new StringBuilder

    report.append("Classification Report\n")
    report.append("=" * 50 + "\n\n")
    report.append(f"Overall Accuracy: ${metrics.accuracy}%.4f\n")
    report.append(f"Overall Precision: ${metrics.precision}%.4f\n")
    report.append(f"Overall Recall: ${metrics.recall}%.4f\n")
    report.append(f"Overall F1 Score: ${metrics.f1}%.4f\n\n")

    report.append("Confusion Matrix:\n")
    metrics.confusionMatrix.foreach { row =>
      report.append(row.map(v => f"$v%.0f").mkString("\t") + "\n")
    }

    report.toString()
  }
}
