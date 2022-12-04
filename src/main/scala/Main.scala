package upm.bd
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

object Main {

  case class FlightSchedule(Year:Int, Month:Int, DayofMonth:Int, DayOfWeek:Int, DepTime:Int,
                            CRSDepTime:String, ArrTime:Int, CRSArrTime:String, UniqueCarrier:String,
                            FlightNum:Int, TailNum:String, ActualElapsedTime:Int, CRSElapsedTime:Int,
                            AirTime:String, ArrDelay:String, DepDelay:String, Origin:String, Dest:String,
                            Distance:Int, TaxiIn:String, TaxiOut:String, Cancelled:Int, CancellationCode:String,
                            Diverted:Int, CarrierDelay:String,WeatherDelay:String, NASDelay:String, SecurityDelay:String,
                            LateAircraftDelay:String)

  case class FlightInfo(flight: Int, duration: String)

  def main(args: Array[String]): Unit = {

    if (args.length > 0) {

      Logger.getLogger("org").setLevel(Level.WARN)

      val spark = SparkSession
        .builder
        .appName("sparkApp")
        .master("local[*]")
        .getOrCreate()

      val datasetPath: String = args(0)

      import spark.implicits._
      val inputDataset = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(datasetPath)

      inputDataset.printSchema()

    }

  }
  //
  //  def makeFlightInfo(schedules: Dataset[FlightSchedule]): Dataset[FlightInfo] = {
  //    implicit val enc: ExpressionEncoder[FlightInfo] = ExpressionEncoder[FlightInfo]
  //    schedules.map(s => existingTransformationFunction(s))
  //  }
  //
  //  def existingTransformationFunction(flightSchedule: FlightSchedule): FlightInfo = {
  //    val duration = (flightSchedule.DepTime - flightSchedule.ArrTime) / 60 / 60
  //    FlightInfo(flightSchedule.FlightNum, s"$duration hrs")
  //  }



}

