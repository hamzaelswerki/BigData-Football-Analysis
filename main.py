import os
import time
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, corr
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

os.environ['JDK_JAVA_OPTIONS'] = '--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.jgss=ALL-UNNAMED'

# During the conducted experiments, the values of instances and cores
# were modified in each experiment in order to analyze
# the impact of different cluster configurations on execution time and speedup and
#and efficiency  and scalability

# Multiple configurations were tested, including:
# - Sequential execution using 1 processing core

# this code to make comparison between Sequential and PARALLEL
#spark = pyspark.sql.SparkSession.builder \
#   .appName("MyFootball Analysis - Sequential Mode") \
#   .master("my loccal") \
#   .getOrCreate()


# And  the following configurations were tested:
:
# - 2 executors × 2 cores
# - 4 executors × 2 cores
# - 8 executors × 2 cores

spark = pyspark.sql.SparkSession.builder \
   .appName("MyFootball Analysis :-> Parallel Experiment 1") \
   .master("spark://10.4.2.149:7077") \
   .config("spark.executor.instances", "8") \
   .config("spark.executor.cores", "2") \
   .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
   .config("spark.executor.memory", "2g") \
   .getOrCreate()
#




file_path = r"C:\Users\MSI\Downloads\football_data.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

df = df.drop("player", "position", "team", "name")
df = df.dropna()

df = df.withColumnRenamed("minutes played", "minutes_played")

# 4. (Feature Engineering)
# إ (Position Flags)
df = df.withColumn("is_goalkeeper", when(df["position_encoded"] == 1, 1).otherwise(0)) \
       .withColumn("is_defender", when(df["position_encoded"] == 2, 1).otherwise(0)) \
       .withColumn("is_midfielder", when(df["position_encoded"] == 3, 1).otherwise(0)) \
       .withColumn("is_attacker", when(df["position_encoded"] == 4, 1).otherwise(0))

df = df.withColumn("attacker_total", (df["goals"] + df["assists"]) * df["is_attacker"])

df = df.withColumn("midfielder_playmaking", (df["assists"] + df["goals"]) * df["is_midfielder"])
df = df.withColumn("defender_defense", (df["clean sheets"] - df["goals conceded"]) * df["is_defender"])
df = df.withColumn("gk_performance", (df["clean sheets"] - df["goals conceded"]) * df["is_goalkeeper"])

# 5.  this code to reomve outliers
df = df.filter(df["current_value"] < 200_000_000)
df = df.filter(df["minutes_played"] > 0)




# this to  test the scalability and load balancing
#df = df.repartition(16)
#print("Data partitions optimized to:", df.rdd.getNumPartitions())

# to avoid recalculating the transformations for each model.
df.cache()
df.count()


# Additional experiments were also conducted to evaluate:
# the effect of increasing dataset size :

# this code to simulation of Scalability by increasing dataset size
#df = df.sample(withReplacement=True, fraction=20.0)


# 7.  (Features Vectorization)
feature_cols = [
    "age", "appearance", "minutes_played", "goals", "assists",
    "days_injured", "games_injured", "award", "position_encoded",
    "attacker_total", "midfielder_playmaking", "defender_defense", "gk_performance"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

#  (Scaling)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# 8.(Data Split)
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 9.(Training)
start = time.time()

# Linear Regression
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="current_value")
lr_model = lr.fit(train)

# Random Forest
rf = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="current_value")
rf_model = rf.fit(train)

end = time.time()
parallel_time = end - start

# 10.  (Evaluation)
evaluator = RegressionEvaluator(labelCol="current_value", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="current_value", predictionCol="prediction", metricName="r2")

#  Linear Regression
lr_preds = lr_model.transform(test)
lr_rmse = evaluator.evaluate(lr_preds)
lr_r2 = evaluator_r2.evaluate(lr_preds)

#  Random Forest
rf_preds = rf_model.transform(test)
rf_rmse = evaluator.evaluate(rf_preds)
rf_r2 = evaluator_r2.evaluate(rf_preds)

print("===========================")
#print(f"Linear Regression RMSE ==> {lr_rmse/1e6:.2f}M")
#print(f"Linear Regression R² ==> {lr_r2:.4f}")
print("===========================")
#print(f"Random Forest RMSE  ==>{rf_rmse/1e6:.2f}M")
#print(f"Random Forest R²  ==> {rf_r2:.4f}")
#print("===========================")
print(f"Processing Time ==> {parallel_time:.2f} seconds")
print("===========================")


print("Partitions:", df.rdd.getNumPartitions())
#print(f"Total Time  ==> {parallel_time:.2f} seconds")


print("===== EXPERIMENT 4 =====")
print("Workers = 4")
print("Executors =", spark.conf.get("spark.executor.instances"))
print("Cores per Executor =", spark.conf.get("spark.executor.cores"))
print("Total Cores =", int(spark.conf.get("spark.executor.instances")) * int(spark.conf.get("spark.executor.cores")))
print("========================")
#print("Row Count:", df.count())


spark.stop()
