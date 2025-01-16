from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def create_spark_session(app_name="MovieRecommender"):
    """Create and configure Spark session"""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "100")
            .config("spark.default.parallelism", "100")
            .getOrCreate())

def load_data(spark, ratings_path, movies_path):
    """Load and prepare MovieLens dataset"""
    # Load ratings data
    ratings_df = spark.read.csv(ratings_path, 
                               header=True, 
                               inferSchema=True)
    
    # Load movies data
    movies_df = spark.read.csv(movies_path,
                              header=True,
                              inferSchema=True)
    
    return ratings_df, movies_df

def prepare_data(ratings_df):
    """Prepare data for ALS model"""
    # Convert to float type
    ratings_df = ratings_df.select(
        col("userId").cast("integer"),
        col("movieId").cast("integer"),
        col("rating").cast("float")
    )
    
    # Split data
    (training, test) = ratings_df.randomSplit([0.8, 0.2], seed=42)
    return training, test

def train_als_model(training_data):
    """Train ALS model with cross-validation"""
    # Initialize ALS
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    
    # Create parameter grid
    param_grid = ParamGridBuilder()\
        .addGrid(als.rank, [10, 50, 100])\
        .addGrid(als.regParam, [0.1, 0.01])\
        .build()
    
    # Define evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    
    # Create cross validator
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )
    
    # Fit model
    model = cv.fit(training_data)
    return model.bestModel

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse

def get_recommendations(model, user_id, movies_df, n=10):
    """Get top N movie recommendations for a user"""
    # Create DataFrame of all movies
    user_df = spark.createDataFrame(
        [[user_id] for _ in range(movies_df.count())],
        ["userId"]
    )
    movie_df = movies_df.select("movieId")
    user_movies = user_df.crossJoin(movie_df)
    
    # Get predictions
    recommendations = model.transform(user_movies)\
        .select("movieId", "prediction")\
        .join(movies_df, "movieId")\
        .orderBy(col("prediction").desc())\
        .limit(n)
    
    return recommendations

if __name__ == "__main__":
    # Initialize Spark
    spark = create_spark_session()
    
    # Load data
    ratings_df, movies_df = load_data(
        spark,
        "ml-20m/ratings.csv",
        "ml-20m/movies.csv"
    )
    
    # Prepare data
    training, test = prepare_data(ratings_df)
    
    # Train model
    model = train_als_model(training)
    
    # Evaluate model
    rmse = evaluate_model(model, test)
    print(f"Root Mean Square Error: {rmse}")
    
    # Example: Get recommendations for user 1
    recs = get_recommendations(model, 1, movies_df)
    recs.show()