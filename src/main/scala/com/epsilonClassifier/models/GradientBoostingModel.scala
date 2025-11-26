package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.{Pipeline, Transformer}
import com.epsilonClassifier.Config

class GradientBoostingModel(config : Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(config.maxIter)
      .setMaxDepth(config.maxDepth)
      .setStepSize(config.learningRate)
      .setSubsamplingRate(0.7)
      .setMinInstancesPerNode(5)
      .setMaxBins(64)
      .setSeed(config.seed)

    val pipeline = new Pipeline().setStages(Array(gbt))
    pipeline.fit(data)
  }
  override def getName: String = "GradientBoosting"
}
