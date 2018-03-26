import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object ModelBasedCF {
    def main(args: Array[String]): Unit = {
        val starttime = System.currentTimeMillis()
        val conf = new SparkConf().setAppName("ModelCF").setMaster("local[2]")
        var sc = new SparkContext(conf)
        var test = sc.textFile(args(1))
        val testheader = test.first()
        test = test.filter(x=>x!=testheader)
//        test = test.filter(row => !row.contains("userID"))
        val test_ratings = test.map(_.split(',')).map(row=> (row(0).toInt, row(1).toInt, row(2).toDouble))



        var total = sc.textFile(args(0))
        total = total.filter(row => !row.contains("userID"))
        val total_ratings = total.map(_.split(',')).map(row => (row(0).toInt, row(1).toInt, row(2).toDouble))
        val ratings = total_ratings.subtract(test_ratings).map(
            info => {
                Rating(info._1, info._2, info._3)
        })
        val rank = 4
        val numIterations = 10
        val lambda = 0.5
        val seed = 1
        val model = ALS.train(ratings, rank, numIterations, lambda, -1, seed)
//        val model = ALS.train(ratings, rank, numIterations,0.01)
        val usersProducts = test_ratings.map (row => {(row._1, row._2)})

        var predictions = model.predict(usersProducts)
                .map { case Rating(user, product, rate) =>((user, product), rate)}

        val min = predictions.map(_._2).min()
        val max = predictions.map(_._2).max()
        var result = predictions.map{case ((user, product), rate) =>
            ((user, product), 4.0 * (rate-min)/(max - min) + 1.0)

        }
        val ratesAndPreds = test_ratings.map { case (user, product, rate) =>
            ((user, product), rate)
        }.join(result)
        val diff = ratesAndPreds.map{case ((user, product), (r1, r2)) =>
        math.abs(r1 - r2)}
        val RMSE = math.sqrt(diff.map(x=>{x * x}).mean())
        var range1 = diff.filter{ case (diff) => diff>=0 && diff<1}.count()
        var range2 = diff.filter{ case (diff) => diff>=1 && diff<2}.count()
        var range3 = diff.filter{ case (diff) => diff>=2 && diff<3}.count()
        var range4 = diff.filter{ case (diff) => diff>=3 && diff<4}.count()
        var range5 = diff.filter{ case (diff) => diff>=4}.count()


        val writer = new PrintWriter(new File(args(2)))
        var out = result.map(row=>{(row._1._1,row._1._2, row._2)})
            .collect().sortBy(x=>(x._1,x._2))
        for (i<-out){
            writer.write(i._1.toString+", "+i._2.toString + ", " + i._3.toString)
            writer.write("\n")
        }
        writer.close()
        println(">=0 and <1: "+range1)
        println(">=1 and <2: "+range2)
        println(">=2 and <3: "+range3)
        println(">=3 and <4: "+range4)
        println(">=4: "+range5)
        println("RMSE: "+ RMSE)
        println("Time: "+ ((System.currentTimeMillis()-starttime)/1000).toString +" sec")

    }
}
