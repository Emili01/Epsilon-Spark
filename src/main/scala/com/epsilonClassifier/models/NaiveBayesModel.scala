package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.{Pipeline, Transformer}
import com.epsilonClassifier.Config


class NaiveBayesModel(config: Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    val nb = new NaiveBayes()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setModelType("multinomial")
        .setSmoothing(2.0)

    val pipeline = new Pipeline().setStages(Array(nb))
    pipeline.fit(data)
  }

  override def getName: String = "NaiveBayes"
}
