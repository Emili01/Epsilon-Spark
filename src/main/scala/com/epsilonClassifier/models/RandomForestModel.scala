package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.{Pipeline, Transformer}
import com.epsilonClassifier.Config

class RandomForestModel(config: Config) extends BaseModel {

  override def train(data: DataFrame): Transformer = {
    val rf = new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setNumTrees(config.numTrees)
        .setMaxDepth(config.maxDepth)
        .setMinInstancesPerNode(5)
        .setMaxBins(64)
        .setSubsamplingRate(0.8)
        .setFeatureSubsetStrategy("sqrt")
        .setSeed(config.seed)

    val pipeline = new Pipeline().setStages(Array(rf))
    pipeline.fit(data)
  }

  override def getName: String = "RandomForest"
}
