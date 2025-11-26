package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{PipelineModel, Transformer}
import com.epsilonClassifier.Config

trait BaseModel {
  def train(data: DataFrame): Transformer
  def getName: String
}

object ModelFactory {
  def getModel(modelName: String, config: Config): BaseModel = {
    modelName.toLowerCase match {
        case "randomforest" | "rf" => new RandomForestModel(config)
        case "logisticregression" | "lr" => new LogisticRegressionModel(config)
        case "naivebayes" => new NaiveBayesModel(config)
        case "mlp" | "multilayerperceptron" => new MLPModel(config)
        case _ => throw new IllegalArgumentException(s"Unknown model: $modelName")
    }
  }
}
