package com.epsilonClassifier.utils

import java.io.{File, PrintWriter}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.MLWritable
import scala.util.{Try, Success, Failure}

class ResultSaver(resultsDir: String, modelName: String, task: String) {

  private val outputDir = s"$resultsDir/${modelName}_$task"

  // Create directory if it doesn't exist
  new File(outputDir).mkdirs()
  new File(s"$outputDir/models").mkdirs()

  def saveMetrics(trainMetrics: Metrics, testMetrics: Metrics): Unit = {
    // Save classification report
    val reportPath = s"$outputDir/classification_report.txt"
    val writer = new PrintWriter(new File(reportPath))

    try {
      writer.write("=" * 60 + "\n")
      writer.write(s"Classification Report: $modelName - $task\n")
      writer.write("=" * 60 + "\n\n")

      writer.write("TRAINING SET METRICS\n")
      writer.write("-" * 60 + "\n")
      writer.write(f"Accuracy:  ${trainMetrics.accuracy}%.4f\n")
      writer.write(f"Precision: ${trainMetrics.precision}%.4f\n")
      writer.write(f"Recall:    ${trainMetrics.recall}%.4f\n")
      writer.write(f"F1 Score:  ${trainMetrics.f1}%.4f\n\n")

      writer.write("Training Confusion Matrix:\n")
      trainMetrics.confusionMatrix.foreach { row =>
        writer.write(row.map(v => f"$v%8.0f").mkString("") + "\n")
      }

      writer.write("\n\nTEST SET METRICS\n")
      writer.write("-" * 60 + "\n")
      writer.write(f"Accuracy:  ${testMetrics.accuracy}%.4f\n")
      writer.write(f"Precision: ${testMetrics.precision}%.4f\n")
      writer.write(f"Recall:    ${testMetrics.recall}%.4f\n")
      writer.write(f"F1 Score:  ${testMetrics.f1}%.4f\n\n")

      writer.write("Test Confusion Matrix:\n")
      testMetrics.confusionMatrix.foreach { row =>
        writer.write(row.map(v => f"$v%8.0f").mkString("") + "\n")
      }

      println(s"Classification report saved to: $reportPath")
    } finally {
      writer.close()
    }

    // Save confusion matrices as CSV
    saveConfusionMatrix(trainMetrics.confusionMatrix, s"$outputDir/train_confusion_matrix.csv")
    saveConfusionMatrix(testMetrics.confusionMatrix, s"$outputDir/test_confusion_matrix.csv")

    // Save metrics summary
    saveMetricsSummary(trainMetrics, testMetrics)
  }

  private def saveConfusionMatrix(matrix: Array[Array[Double]], path: String): Unit = {
    val writer = new PrintWriter(new File(path))
    try {
      matrix.foreach { row =>
        writer.write(row.mkString(",") + "\n")
      }
      println(s"Confusion matrix saved to: $path")
    } finally {
      writer.close()
    }
  }

  private def saveMetricsSummary(trainMetrics: Metrics, testMetrics: Metrics): Unit = {
    val path = s"$outputDir/metrics_summary.csv"
    val writer = new PrintWriter(new File(path))
    try {
      writer.write("split,accuracy,precision,recall,f1\n")
      writer.write(f"train,${trainMetrics.accuracy}%.6f,${trainMetrics.precision}%.6f,${trainMetrics.recall}%.6f,${trainMetrics.f1}%.6f\n")
      writer.write(f"test,${testMetrics.accuracy}%.6f,${testMetrics.precision}%.6f,${testMetrics.recall}%.6f,${testMetrics.f1}%.6f\n")
      println(s"Metrics summary saved to: $path")
    } finally {
      writer.close()
    }
  }

  def saveModel(model: Transformer, modelName: String): Unit = {
    val modelPath = s"$outputDir/models/$modelName"
    model match {
      case writable: MLWritable =>
        Try {
          writable.write.overwrite().save(modelPath)
        } match {
          case Success(_) =>
            println(s"Model saved to: $modelPath")
          case Failure(e) =>
            println(s"Warning: Could not save model: ${e.getMessage}")
        }
      case _ =>
        println(s"Warning: Model of type ${model.getClass.getName} is not saveable.")
    }
  }
}
