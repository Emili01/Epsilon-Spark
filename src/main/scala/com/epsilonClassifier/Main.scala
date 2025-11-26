package com.epsilonClassifier // Asegura que coincida con tu carpeta

import org.apache.spark.sql.SparkSession
import scopt.OParser
import com.epsilonClassifier.models._
import com.epsilonClassifier.utils._

case class Config (
    model: String = "RandomForest",
    task: String = "epsilon", // Cambiado por defecto a epsilon
    trainPath: String = "epsilon_train.parquet", // Rutas por defecto para tu cluster
    testPath: String = "epsilon_test.parquet",
    resultsDir: String = "results",
    maxIter: Int = 100,
    maxDepth: Int = 10,
    numTrees: Int = 100,
    learningRate: Double = 0.1,
    regParam: Double = 0.01,
    elasticNetParam: Double = 0.0,
    seed : Int = 42
)

object Main {
  def main(args: Array[String]): Unit = {
    val builder = OParser.builder[Config]
    val parser = {
      import builder._
      OParser.sequence(
        programName("EpsilonClassifier"),
        head("Epsilon Classifier", "1.0"),
        opt[String]('m', "model")
          .action((x, c) => c.copy(model = x))
          .text("Model: RandomForest, LogisticRegression, GradientBoosting, MLP"),
        opt[String]('t', "task")
          .action((x, c) => c.copy(task = x))
          .text("Task: epsilon (default), category, sentiment"),
        opt[String]("train")
          .action((x, c) => c.copy(trainPath = x))
          .text("Path to training data (Parquet)"),
        opt[String]("test")
          .action((x, c) => c.copy(testPath = x))
          .text("Path to test data (Parquet)"),
        opt[String]("results")
          .action((x, c) => c.copy(resultsDir = x))
          .text("Results directory"),
        // ... (resto de opciones igual) ...
        opt[Int]("max-iter").action((x, c) => c.copy(maxIter = x)),
        opt[Int]("max-depth").action((x, c) => c.copy(maxDepth = x)),
        opt[Int]("num-trees").action((x, c) => c.copy(numTrees = x)),
        opt[Double]("lr").action((x, c) => c.copy(learningRate = x)),
        opt[Int]("seed").action((x, c) => c.copy(seed = x))
      )
    }

    OParser.parse(parser, args, Config()) match {
      case Some(config) => run(config)
      case _ => sys.exit(1)
    }
  }

  def run(config: Config): Unit = {
    // CLUSTER CONFIGURATION:
    // No definimos .master() aquí. Se define al ejecutar spark-submit.
    // No definimos memoria fija aquí. Se define con --driver-memory y --executor-memory.
    val spark = SparkSession
      .builder()
      .appName(s"Epsilon_${config.model}")
      // Optimizaciones para Epsilon (Dataset denso y grande)
      .config("spark.sql.parquet.enableVectorizedReader", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.shuffle.partitions", "200") // Ajustar según el tamaño del cluster
      .getOrCreate()

    try {
      println(s"=== Cluster Job Started: ${config.model} on ${config.task} ===")
      println(s"Configuration: $config")

      val dataLoader = new DataLoader(spark)
      val (trainDf, testDf) = dataLoader.loadData(config.trainPath, config.testPath, config.task)

      println(s"Training samples: ${trainDf.count()}")
      println(s"Test samples: ${testDf.count()}")

      val model = ModelFactory.getModel(config.model, config)

      println("\n=== Training model ===")
      val trainedModel = model.train(trainDf)

      println("\n=== Evaluating model ===")
      val evaluator = new Evaluator(spark, config.task)
      val trainMetrics = evaluator.evaluate(trainedModel, trainDf, "train")
      val testMetrics = evaluator.evaluate(trainedModel, testDf, "test")

      val resultSaver = new ResultSaver(config.resultsDir, config.model, config.task)
      resultSaver.saveMetrics(trainMetrics, testMetrics)
      
      // Guardar modelo en HDFS o ruta compartida
      resultSaver.saveModel(trainedModel, s"${config.model}_${config.task}")

    } finally {
      spark.stop()
    }
  }
}