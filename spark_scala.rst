Spark SQL
################################################

Overview
================================================
Apache Spark is a fast and general-purpose cluster computing system.
It provides high-level APIs in Java, Scala, Python and R,
and an optimized engine that supports general execution graphs.
It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing,
MLlib for machine learning, GraphX for graph processing, and Spark Streaming.

Spark SQL is a Spark module for structured data processing.
Unlike the basic Spark RDD API, the interfaces provided by Spark SQL provide Spark with more information about the structure of both the data and the computation being performed.
Internally, Spark SQL uses this extra information to perform extra optimizations.

SQL
================================================
One use of Spark SQL is to execute SQL queries.
Spark SQL can also be used to read data from an existing Hive installation.
When running SQL from within another programming language the results will be returned as a Dataset/DataFrame.

Datasets and DataFrames
================================================

1. A Dataset is a distributed collection of data.

  * A Dataset can be constructed from JVM objects and then manipulated using functional transformations(map, flatMap, filter, etc.).
  * The Dataset API is avaiable in Scala and Java. 

2. A DataFrame is a Dataset organized into named columns. 

   * Dataframes can be constructed from a wide array of sources such as:
     structed data files, tables in Hive, external databases, or existing RDDs.
   * In Scala and Java, a DataFrame is represented by a Dataset of Rows.
   * In the Scala API, DataFrame is simply a type alias of Dataset[Row]

3. Starting point: SparkSession

   * SparkSession in Spark 2.0 provieds builtin support for Hive features including the ability to wirte queries using HiveQL,
     access to Hive UDFs, and the ability to read data from Hive tables.

4. Creating DataFrames

   * With a SparkSession, applications can create DataFrames from an existing RDD, from a Hive table, or from Spark data sources.

Data Sources
================================================

Spark SQL supports operating on a variety of data sources through the DataFrame interfance.
A DataFrame can be operated on using relational transformations and can also be used to create a temporary view.

SparkContext and SparkConf and SparkSession
================================================

1. SparkConf is a class.

   * It sets configuration for a Spark application.
   * Used to set various Spark parameters as key-value paris.
   * Once a SparkConf object is passed to Spark, it is cloned and can no longer be modified by the user.
   * Spark does not support modifying the configuration at runtime.

2. SparkSession is a class.
   
   * it is introduced in Spark 2.0. It is a new entry point that subsumes SparkContext, SQLContext, StreamingContext, and HiveContext to progrmming Spark with the Dataset and DateFrame API.
   * To create a basic SparkSession, just use SparkSession.builder().
   * The builder will automatically reuse an existing SparkContext if one exists; and create a SparkContext if it does not exist.
   * SparkSession is the entry point for reading data, similar to the old SQLContext.read.

3. SparkContext is a class.

   * it is the main entry point for Spark functionality.
   * A SparkContext represents the connection to a Spark cluster, and can be used to create RDDs, accumulators and broadcast variables on that cluster.

4. The sql function on a SparkSession enables applications to run SQL queries programmatically and returns the result as a DataFrame.

Creating Datasets
================================================

Datasets are similar to RDDs, however, instead of using Java serialization of Kryo they use a specialized Encoder to serialize the objects for processing or transmitting over the network.
While both encoders and standard serialization are responsible for turning an object into bytes,
encoders are code generated dynamically and use a format that allows Spark to perform many operations like filtering,
sorting and hashing without deserializing the bytes back into an object.


Read and Write files from Hive
################################################

Common part
================================================

sbt Dependencies
------------------------------------------------

.. code:: scala

  libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % "proviede"
  libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0" % "proviede"

assembly Dependencies
------------------------------------------------

.. code:: scala

  // In build.sbt
  import sbt.Keys._
  assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

  // in project/assembly.sbt
  addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.1")

note on sbt-assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. sbt-assembly is a sbt plugin originally ported from codahale's assembly-sbt.
The goal is simple: Create a fat JAR of your project with all of its dependencies.

Init SparkSession
================================================

.. code:: scala

  // Creation of SparkSession
  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("some name")
    .config("spark.some.config.option", "some-value")
    .enableHiveSuport()
    .getOrCreate()

 // For implicit conversions like converting RDDs to DataFrames
 import spark.implicits._


SparkSession provides builtin support for Hive features including the ability to write queries using HiveQL, access to Hive UDFs, and the ability to read data from Hive tables.

How to write to a Hive table with Spark Scala?
================================================

.. code:: scala

  // ======= Creating a dataframe with 1 partition
  import sparkSession.implicits._
  val df = Seq(HelloWorld("helloworld")).toDF().coalesce(1)

  // ======= Writing files
  // Writing Dataframe as a Hive table
  import sparkSession.sql
  sql("DROP TABLE IF EXISTS helloworld")
  sql("CREATE TABLE helloworld (message STRING)")
  df.write.mode(SaveMode.Overwrite).saveAsTable("helloworld")
  logger.info("Writing hive table: OK")

How to read from a Hive table with Spark Scala?
================================================

.. code:: scala

  // ======= Reading files
  // Reading hive table into a Spark Dataframe
  val dfHive = sql("SELECT * from helloworld")
  logger.info("Reading hive table: OK")
  logger.info(dfHive.show())

How to write to a Hive table with Spark Scala?
================================================

.. code:: scala

  // Defining a Helloworld class
  case class HelloWorld(messaage: String)
  // ======= Creating a dataframe with 1 partition
  val df = Seq(HelloWorld("helloworld")).toDF().coalesce(1)

  // ======= Writing files
  // Writing Dataframe as parquet file
  df.write.mode(SaveMode.Overwrite).parquet(hdfs_master + "user/hdfs/wiki/testwiki")
  // Writing Dataframe as csv file
  df.write.mode(SaveMode.Overwrite).csv(hdfs_master + "user/hdfs/wiki/testwiki.csv")

How to read from HDFS with Spark Scala?
================================================

.. code:: scala

  // ======= Reading files
  // Reading parquet files into a Spark Dataframe
  val df_parquet = session.read.parquet(hdfs_master + "user/hdfs/wiki/testwiki")
  // Reading csv files into a Spark Dataframe
  val df_csv = sparkSession.read.option("inferSchema", "true").csv(hdfs_master + "user/hdfs/wiki/testwiki.csv")


How to read from a Hive table with Spark Scala?
================================================

Spark APIs: RDDs, DataFrames, and Datasets
================================================


