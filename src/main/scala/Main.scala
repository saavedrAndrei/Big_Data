package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler


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

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Year",
        "Month",
        "DayofMonth",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "FlightNum",
        "CRSElapsedTime",
        "DepDelay",
        "Distance"
      ))
      .setOutputCol("features")

    val df = assembler.transform(subset3)
      .select("label", "features")

    println(df)

    val trainTest = df.randomSplit(Array(0.8, 0.2))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)

    println(trainingDF)
    println(testDF)

    val linear = new LinearRegression()
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(100)
      .setTol(1E-6)

    println(linear)
    val model = linear.fit(trainingDF)
    println(model)

//    val fullPredictions =  model.transform(testDF).cache()
//
//    println(fullPredictions)

//    val evaluation = fullPredictions.select("prediction", "ArrDelay").collect()
//
//    for (prediction <- evaluation) {
//      println(prediction)
//    }


    spark.stop()
  }

}

