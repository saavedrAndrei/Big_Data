package upm.bd
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types._



object Main {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    val spark = SparkSession
      .builder
      .appName("sparkApp")
      .master("local[*]")
      .getOrCreate()

    val customSchema = StructType(Array(
      StructField("Year", IntegerType, true),
      StructField("Month", IntegerType, true),
      StructField("DayofMonth", IntegerType, true),
      StructField("DayofWeek", IntegerType, true),
      StructField("DepTime", IntegerType, true),
      StructField("CRSDepTime", IntegerType, true),
      StructField("ArrTime", IntegerType, true),
      StructField("CRSArrTime", IntegerType, true),
      StructField("UniqueCarrier", StringType, true),
      StructField("FlightNum", IntegerType, true),
      StructField("TailNum", StringType, true),
      StructField("ActualElapsedTime", IntegerType, true),
      StructField("CRSElapsedTime", IntegerType, true),
      StructField("AirTime", StringType, true),
      StructField("ArrDelay", IntegerType, true),
      StructField("DepDelay", IntegerType, true),
      StructField("Origin", StringType, true),
      StructField("Dest", StringType, true),
      StructField("Distance", IntegerType, true),
      StructField("TaxiIn", IntegerType, true),
      StructField("TaxiOut", IntegerType, true),
      StructField("Cancelled", IntegerType, true),
      StructField("CancellationCode", IntegerType, true),
      StructField("Diverted", IntegerType, true),
      StructField("CarrierDelay", IntegerType, true),
      StructField("WeatherDelay", IntegerType, true),
      StructField("NASDelay", IntegerType, true),
      StructField("SecurityDelay", IntegerType, true),
      StructField("LateAircraftDelay", IntegerType, true))
    )

    import spark.implicits._
    val inputDataset = spark.read
      .option("header", "true")
      .schema(customSchema)
      .csv("data/1987-1.csv")

    inputDataset.printSchema()

    val subset1 = inputDataset
      .filter("Cancelled == 0")
      .select(
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
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

    subset1.printSchema()
    subset1.show()

  }

}

