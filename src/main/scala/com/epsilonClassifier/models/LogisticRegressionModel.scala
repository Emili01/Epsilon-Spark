package com.epsilonClassifier.models

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, Transformer}
import com.epsilonClassifier.Config

class LogisticRegressionModel(config: Config) extends BaseModel {
  override def train(data: DataFrame): Transformer = {
    val lr = new LogisticRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setMaxIter(config.maxIter * 2)
        .setRegParam(config.regParam * 0.5)
        .setElasticNetParam(0.8) // More L1
        .setFamily("multinomial")
        .setTol(1e-6)

    val pipeline = new Pipeline().setStages(Array(lr))
    pipeline.fit(data)
  }

  override def getName: String = "LogisticRegression"
}
