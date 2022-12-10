package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row
import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.StandardScaler

object Prueba {

  case class dataSchema(Year:Int, Month:Int, DayofMonth:Int, DayOfWeek:Int, DepTime:Int,
                        CRSDepTime:Int, ArrTime:Int, CRSArrTime:Int, UniqueCarrier:String,
                        FlightNum:Int, TailNum: String, ActualElapsedTime:Int, CRSElapsedTime:Int,
                        AirTime: String, ArrDelay:Double, DepDelay: Int, Origin:String, Dest:String,
                        Distance: Int, TaxiIn:Int, TaxiOut:Int, Cancelled: Int, CancellationCode:Int,
                        Diverted: Int, CarrierDelay:Int, WeatherDelay:Int, NASDelay:Int, SecurityDelay:Int,
                        LateAircraftDelay:Int)

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("sparkApp")
      .master("local[*]")
      .getOrCreate()

    val customSchema = new StructType()
      .add("Year", IntegerType, true)
      .add("Month", IntegerType, true)
      .add("DayofMonth", IntegerType, true)
      .add("DayOfWeek", IntegerType, true)
      .add("DepTime", IntegerType, true)
      .add("CRSDepTime", IntegerType, true)
      .add("ArrTime", IntegerType, true)
      .add("CRSArrTime", IntegerType, true)
      .add("UniqueCarrier", StringType, true)
      .add("FlightNum", IntegerType, true)
      .add("TailNum", StringType, true)
      .add("ActualElapsedTime", IntegerType, true)
      .add("CRSElapsedTime", IntegerType, true)
      .add("AirTime", StringType, true)
      .add("ArrDelay", DoubleType, true)
      .add("DepDelay", IntegerType, true)
      .add("Origin", StringType, true)
      .add("Dest", StringType, true)
      .add("Distance", IntegerType, true)
      .add("TaxiIn", IntegerType, true)
      .add("TaxiOut", IntegerType, true)
      .add("Cancelled", IntegerType, true)
      .add("CancellationCode", IntegerType, true)
      .add("Diverted", IntegerType, true)
      .add("CarrierDelay", IntegerType, true)
      .add("WeatherDelay", IntegerType, true)
      .add("NASDelay", IntegerType, true)
      .add("SecurityDelay", IntegerType, true)
      .add("LateAircraftDelay", IntegerType, true)

    import spark.implicits._
    val inputDataset = spark.read
      .option("header", "true")
      .schema(customSchema)
      .csv("data/1987-11.csv")
      .as[dataSchema]

    val subset1 = inputDataset
      .filter("Cancelled == 0")
      .select(
        "Year",
        "Month",
        "DayofMonth",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "CRSElapsedTime",
        "ArrDelay",
        "DepDelay",
        "Distance",
        "Cancelled"
      )
    val subset2 = subset1.drop("Cancelled")

    val numerical_imputer = new Imputer()
      .setInputCols(Array(
        "Year",
        "Month",
        "DayofMonth",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "ArrDelay",
        "DepDelay",
        "Distance",
      ))
      .setOutputCols(Array(
        "Year_imputed",
        "Month_imputed",
        "DayofMonth_imputed",
        "DepTime_imputed",
        "CRSDepTime_imputed",
        "CRSArrTime_imputed",
        "CRSElapsedTime_imputed",
        "ArrDelay_imputed",
        "DepDelay_imputed",
        "Distance_imputed",
      ))
      .setStrategy("mean")

    val model_numerical_imputer = numerical_imputer.fit(subset2).transform(subset2)
    val cleaning_01 = model_numerical_imputer.drop(
      "Year",
      "Month",
      "DayofMonth",
      "DepTime",
      "CRSDepTime",
      "CRSArrTime",
      "CRSElapsedTime",
      "ArrDelay",
      "DepDelay",
      "Distance",
    )
    val model_categorical_imputer = cleaning_01.na.drop()


    val indexer = new StringIndexer()
      .setInputCol("UniqueCarrier")
      .setOutputCol("UniqueCarrierIndex")
    val encoder = new OneHotEncoder()
      .setInputCol("UniqueCarrierIndex")
      .setOutputCol("UniqueCarrier_vector")
    val pipeline_preprocess = new Pipeline()
      .setStages(Array(
        indexer,
        encoder))

    val model_preprocessed = pipeline_preprocess.fit(model_categorical_imputer).transform(model_categorical_imputer)

    val cleaning_02 = model_preprocessed.drop("UniqueCarrier")

    val subset4 = cleaning_02.withColumnRenamed("ArrDelay_imputed","label")

    val trainTest = subset4.randomSplit(Array(0.8, 0.2))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)

    def onlyFeatureCols(c: String): Boolean = !(c matches "label")
    val featureCols = trainingDF.columns
      .filter(onlyFeatureCols)
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    val linear = new LinearRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(100)
      .setTol(1E-6)
    val pipeline_model = new Pipeline()
      .setStages(Array(
        assembler,
        scaler,
        linear))
    val paramGrid = new ParamGridBuilder()
      .addGrid(linear.regParam, Array(0.1, 0.01))
      .addGrid(linear.fitIntercept)
      .addGrid(linear.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline_model)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val cvModel = cv.fit(trainingDF)

    val trainPredictionsAndLabels = cvModel.transform(trainingDF).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val testPredictionsAndLabels = cvModel.transform(testDF).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val testRegressionMetrics = new RegressionMetrics(testPredictionsAndLabels)

    val log = LogManager.getRootLogger

    val results = cvModel.transform(testDF)
      .select("label", "prediction")
      .collect()
      .foreach { case Row(label: Double, prediction: Double) =>
        println(s"--> label=$label, prediction=$prediction")
      }

    log.info(results)

    val output = "\n=====================================================================\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      "=====================================================================\n" +
      s"Test data RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
      "=====================================================================\n"

    log.info(output)

    //    println("\n=================4====================================================\n")
//    subset4.show()
//    println("=====================================================================\n")


//
//    val trainTest = subset3.randomSplit(Array(0.8, 0.2))
//    val trainingDF = trainTest(0)
//    val testDF = trainTest(1)
//
//
//
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(linear.regParam, Array(0.1, 0.01))
//      .addGrid(linear.fitIntercept)
//      .addGrid(linear.elasticNetParam, Array(0.0, 0.5, 1.0))
//      .build()
//
//    val df_transformed = pipeline.fit(trainingDF).transform(trainingDF)
//    println("\n=====================================================================\n")
//    df_transformed.show()
//    println("=====================================================================\n")
//
//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(new RegressionEvaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(2)
//
//    val cvModel = cv.fit(df_transformed)
//
////    val resultModels = cvModel.transform(df_transformed)
//
//
//    val trainPredictionsAndLabels = cvModel.transform(df_transformed).select("label", "prediction")
//      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
//
//    val testPredictionsAndLabels = cvModel.transform(testDF).select("label", "prediction")
//      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
//
//    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
//    val testRegressionMetrics = new RegressionMetrics(testPredictionsAndLabels)
//
//    val log = LogManager.getRootLogger
//
//    val results = cvModel.transform(testDF)
//      .select("label", "prediction")
//      .collect()
//      .foreach { case Row(label: Double, prediction: Double) =>
//        println(s"--> label=$label, prediction=$prediction")
//      }
//
//    log.info(results)
//
//    val output = "\n=====================================================================\n" +
//      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
//      "=====================================================================\n" +
//      s"Test data RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
//      "=====================================================================\n"
//
//    log.info(output)
//    log.info(featureCols)

    spark.stop()
  }

}