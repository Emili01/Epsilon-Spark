package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.{Pipeline, Transformer}
import com.epsilonClassifier.Config

class MLPModel(config: Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    val featureDim = data.select("features").first()
      .getAs[org.apache.spark.ml.linalg.Vector](0).size

    val numClasses = data.select("label").distinct().count().toInt

    val layers = Array(featureDim, 256, 128, 64, numClasses)

    val mlp = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setLayers(layers)
      .setMaxIter(config.maxIter)
      .setBlockSize(128)
      .setSeed(config.seed)
      .setStepSize(0.03)

    val pipeline = new Pipeline().setStages(Array(mlp))
    pipeline.fit(data)
  }

  override def getName: String = "MLP"
}
