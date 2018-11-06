Spark SQL
################################################

Overview
================================================

Spark SQL is a Spark module for structured data processing.
One use of Spark SQL is to execute SQL queries.
Spark SQL can also be used to read data from an existing Hive installation.
When running SQL from within another programming language the results will be returned as a Dataset/DataFrame.

Datasets and DataFrames
================================================

1. A Dataset is a distributed collection of data.

  * A Dataset can be constructed from JVM objects and then manipulated using functional transformations(map, flatMap, filter, etc.).
  * The Dataset API is avaiable in Scala and Java. 

2. A DataFrame is a Dataset organized into named columns. 

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

Read and Write files from HDFS
################################################

Read and Write files from MongaDB
################################################



