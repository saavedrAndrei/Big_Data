package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression,RandomForestRegressor}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row
import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType


object Prueba {


  trait DelegatingEstimatorModelParams extends Params {
    final val selectedEstimator = new Param[Int](this, "selectedEstimator", "The selected estimator")
  }

  class DelegatingEstimator private(override val uid: String, delegates: Array[Estimator[_]]) extends Estimator[DelegatingEstimatorModel] with DelegatingEstimatorModelParams {
    private def this(estimators: Array[Estimator[_]]) = this(Identifiable.randomUID("delegating-estimator"), estimators)

    def this(estimator1: Estimator[_], estimator2: Estimator[_], estimators: Estimator[_]*) = {
      this((Seq(estimator1, estimator2) ++ estimators).toArray)
    }

    setDefault(selectedEstimator -> 0)

    override def fit(dataset: Dataset[_]): DelegatingEstimatorModel = {
      val estimator = delegates(getOrDefault(selectedEstimator))
      val model = estimator.fit(dataset).asInstanceOf[Model[_]]
      new DelegatingEstimatorModel(uid, model)
    }

    override def copy(extra: ParamMap): Estimator[DelegatingEstimatorModel] = {
      val that = new DelegatingEstimator(uid, delegates)
      copyValues(that, extra)
    }

    override def transformSchema(schema: StructType): StructType = {
      // All delegates are assumed to perform the same schema transformation,
      // so we can simply select the first one:
      delegates(0).transformSchema(schema)
    }
  }

  class DelegatingEstimatorModel(override val uid: String, val delegate: Model[_]) extends Model[DelegatingEstimatorModel] with DelegatingEstimatorModelParams {
    def copy(extra: ParamMap): DelegatingEstimatorModel = new DelegatingEstimatorModel(uid, delegate.copy(extra).asInstanceOf[Model[_]])

    def transform(dataset: Dataset[_]): DataFrame = delegate.transform(dataset)

    def transformSchema(schema: StructType): StructType = delegate.transformSchema(schema)
  }

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
      .csv("data/1987.csv")
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
        "TaxiOut",
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

    cleaning_02.printSchema()
    cleaning_02.show()

    val subset4 = cleaning_02.withColumnRenamed("ArrDelay_imputed","label")

    subset4.createOrReplaceTempView("view")
    val subset5 = spark.sql(
      "select Year_imputed, Month_imputed, DayofMonth_imputed,INT(substring(lpad(DepTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(DepTime_imputed,4,0), 3, 2)) " +
        "as DepTime_conv,INT(substring(lpad(CRSDepTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(CRSDepTime_imputed,4,0), 3, 2)) as " +
        "CRSDepTime_conv, INT(substring(lpad(CRSArrTime_imputed,4,0), 1, 2))*60+INT(substring(lpad(CRSArrTime_imputed,4,0), 3, 2)) as " +
        "CRSArrTime_conv,CRSElapsedTime_imputed, label, DepDelay_imputed, Distance_imputed, UniqueCarrierIndex, UniqueCarrier_vector from view"
    )

    subset5.printSchema()
    subset5.show()

    val trainTest = subset5.randomSplit(Array(0.8, 0.2))
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
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
    val pipeline_model = new Pipeline()
      .setStages(Array(
        assembler,
        scaler,
        linear))

    val delegatingEstimator = new DelegatingEstimator(linear, rf)

    val paramGrid = new ParamGridBuilder()
      .addGrid(delegatingEstimator.selectedEstimator, Array(0, 1))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline_model)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val cvModel = cv.fit(trainingDF)

    val bestModel = cvModel.bestModel.asInstanceOf[DelegatingEstimatorModel].delegate

    val log = LogManager.getRootLogger
    log.info(bestModel)

    spark.stop()
  }

}