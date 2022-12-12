package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row
import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.StandardScaler
import upm.bd.Prueba.dataSchema

object Main {

  case class dataSchema(Year:Int, Month:Int, DayofMonth:Int, DayOfWeek:Int, DepTime:Int,
                            CRSDepTime:Int, ArrTime:Int, CRSArrTime:Int, UniqueCarrier:String,
                            FlightNum:Int, TailNum: String, ActualElapsedTime:Int, CRSElapsedTime:Int,
                            AirTime: String, ArrDelay:Double, DepDelay: Int, Origin:String, Dest:String,
                            Distance: Int, TaxiIn:Int, TaxiOut:Int, Cancelled: Int, CancellationCode:Int,
                            Diverted: Int, CarrierDelay:Int, WeatherDelay:Int, NASDelay:Int, SecurityDelay:Int,
                            LateAircraftDelay:Int)

  def main(args: Array[String]): Unit = {

    if (args.length > 0) {
      Logger.getLogger("org").setLevel(Level.ERROR)

      val spark = SparkSession
        .builder
        .appName("sparkApp")
        .master("local[*]")
        .getOrCreate()

      // DEFINE THE CUSTOM SCHEMA BY CHANGING THEIR TYPE
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

      // READ THE DATASET AND APPLY THE SCHEMA PREDEFINED
      val datasetPath: String = args(0)
      import spark.implicits._
      val inputDataset = spark.read
        .option("header", "true")
        .schema(customSchema)
        .csv(datasetPath)
        .as[dataSchema]

      // FILTERED OUT ALL FLIGHTS THAT WERE CANCELLED
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
          "TaxiOut",
          "Cancelled"
        )

      // DROP THE VARIABLE CANCELLED
      val subset2 = subset1.drop("Cancelled")

      // IMPUTATION OF NUMERICAL MISSING VALUES WITH THE MEAN
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
          "TaxiOut"
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
          "TaxiOut_imputed"
        ))
        .setStrategy("mean")
      val model_numerical_imputer = numerical_imputer.fit(subset2).transform(subset2)

      // DROP THE OLD VARIABLES THAT WERE NOT IMPUTED
      val cleaning_01 = model_numerical_imputer.drop(
        "Year",
        "Month",
        "DayofMonth",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "label",
        "DepDelay",
        "Distance",
        "TaxiOut"
      )
      val model_categorical_imputer = cleaning_01.na.drop()

      // ONE HOT ENCODING CATEGORICAL VARIABLE: UniqueCarrier
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

      // EXECUTE THE PIPELINE FOR PREPROCESSING
      val model_preprocessed = pipeline_preprocess.fit(model_categorical_imputer).transform(model_categorical_imputer)

      // DROP THE ORIGINAL VARIABLE IN STRING FORMAT
      val cleaning_02 = model_preprocessed.drop("UniqueCarrier")

      // RENAMED THE TARGET VARIABLE ArrDelay_imputed TO label
      val subset4 = cleaning_02.withColumnRenamed("ArrDelay_imputed", "label")

      // TRANSFORM DepTime, CRSDepTime and CRSArrTime TO MINUTES AND FILTER ONLY NECESSARY VARIABLES
      subset4.createOrReplaceTempView("view")
      val subset5 = spark.sql(
        "select Year_imputed, Month_imputed, DayofMonth_imputed,INT(substring(lpad(DepTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(DepTime_imputed,4,0), 3, 2)) " +
          "as DepTime_conv,INT(substring(lpad(CRSDepTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(CRSDepTime_imputed,4,0), 3, 2)) as " +
          "CRSDepTime_conv, INT(substring(lpad(CRSArrTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(CRSArrTime_imputed,4,0), 3, 2)) as " +
          "CRSArrTime_conv,CRSElapsedTime_imputed, label, DepDelay_imputed, Distance_imputed, UniqueCarrierIndex, UniqueCarrier_vector from view"
      )

      // SPLIT DATASET INTO TRAIN AND TEST
      val trainTest = subset5.randomSplit(Array(0.8, 0.2))
      val trainingDF = trainTest(0)
      val testDF = trainTest(1)

      // SELECT ALL FEATURES VARIABLES BY EXCLUDING label
      def onlyFeatureCols(c: String): Boolean = !(c matches "label")


      // APPLY THE PIPELINE TO FEED THE MODEL
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

      // HYPER-PARAMETER TUNNING THROUGH CROSS VALIDATION TECHNIQUE
      val cv = new CrossValidator()
        .setEstimator(pipeline_model)
        .setEvaluator(new RegressionEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2)

      // EXECUTE THE CROSS VALIDATION ON TRAINING SPLIT
      val cvModel = cv.fit(trainingDF)


      // EVALUATE THE RESULTS
      val trainPredictionsAndLabels = cvModel.transform(trainingDF).select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

      val testPredictionsAndLabels = cvModel.transform(testDF).select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

      val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
      val testRegressionMetrics = new RegressionMetrics(testPredictionsAndLabels)

      val log = LogManager.getRootLogger

      // SHOW THE PREDICTIONS AND LABELS
      val results = cvModel.transform(testDF)
        .select("label", "prediction")
        .collect()
        .foreach { case Row(label: Double, prediction: Double) =>
          println(s"--> label=$label, prediction=$prediction")
        }

      log.info(results)

      // PRINT THE METRICS
      val output = "\n=====================================================================\n" +
        s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
        s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
        s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
        "=====================================================================\n" +
        s"Test data RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
        s"Test data R-squared = ${testRegressionMetrics.r2}\n" +
        s"Test data Explained variance = ${testRegressionMetrics.explainedVariance}\n" +
        "=====================================================================\n"

      log.info(output)

      spark.stop()
    }
  }

}