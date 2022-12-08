package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row
import org.apache.log4j.LogManager
object Main {

  case class dataSchema(Year:Int, Month:Int, DayofMonth:Int, DayOfWeek:Int, DepTime:Int,
                            CRSDepTime:Int, ArrTime:Int, CRSArrTime:Int, UniqueCarrier:String,
                            FlightNum:Int, TailNum: String, ActualElapsedTime:Int, CRSElapsedTime:Int,
                            AirTime: String, ArrDelay:Int, DepDelay: Int, Origin:String, Dest:String,
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
      .add("ArrDelay", IntegerType, true)
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
      .csv("data/1987-1.csv")
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
        "FlightNum",
        "CRSElapsedTime",
        "ArrDelay",
        "DepDelay",
        "Distance",
        "Cancelled"
      )

    val subset2 = subset1.drop("Cancelled")
    val subset3 = subset2.withColumnRenamed("ArrDelay","label")


    val subset4 = subset3.na.drop()

    val trainTest = subset4.randomSplit(Array(0.8, 0.2))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)

    def onlyFeatureCols(c: String): Boolean = !(c matches "label")
    val featureCols = trainingDF.columns
      .filter(onlyFeatureCols)
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    val linear = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(100)
      .setTol(1E-6)
    val pipeline = new Pipeline()
      .setStages(Array(assembler,linear))

    val lr = new LinearRegression()
      .setMaxIter(10)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(trainingDF)

    val trainPredictionsAndLabels = cvModel.transform(trainingDF).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val testPredictionsAndLabels = cvModel.transform(testDF).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val testRegressionMetrics = new RegressionMetrics(testPredictionsAndLabels)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val log = LogManager.getRootLogger

    val output = "\n=====================================================================\n" +
      "=====================================================================\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      "=====================================================================\n" +
      s"Test data RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
      "=====================================================================\n" +
      s"Best Model = ${bestModel}"

    log.info(output)
//    val predictions =  model.transform(testDF).cache()
//
//    val evaluation=predictions.select("prediction","label").collect()
//
//    for(prediction<-evaluation){
//    println(prediction)
//    }

    spark.stop()
  }

}

