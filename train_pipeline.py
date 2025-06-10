import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd


spark = SparkSession.builder \
    .appName("MLflowTask") \
    .getOrCreate()


data_pd = pd.DataFrame({
    'input1': [1, 2, 3, 4, 5, 6],
    'input2': [10, 20, 30, 40, 50, 60],
    'label': [0, 0, 1, 0, 1, 1]
})

data = spark.createDataFrame(data_pd)

assembler = VectorAssembler(inputCols=["input1", "input2"], outputCol="features")
data = assembler.transform(data).select("features", "label")

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

mlflow.set_experiment("spark-mlflow-logreg")

with mlflow.start_run():
    
    model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_data)

    try:
        mlflow.spark.log_model(model, "logistic-model")
    except Exception as e:
        raise  

 
    predictions = model.transform(test_data)
    if predictions.count() > 0:
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        accuracy = evaluator.evaluate(predictions)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No test predictions to evaluate.")


spark.stop()
