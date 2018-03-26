
import org.apache.spark.{SparkConf, SparkContext}
import java.io._


object UserBasedCF {
    def correlation (u_p_list: Array[(Int, Set[Int])], u_p_r_map: Map[Int, Map[Int, Double]], min_intersection: Int): Map[(Int,Int), Double] ={
        val nrows = u_p_list.size
        // pearson corr map
        var corrMap = Map.empty[(Int, Int), Double]
        for (i<- 0 to nrows -1){
            for (j<-i+1 to nrows -1){
                var mutual = u_p_list(i)._2.intersect(u_p_list(j)._2)
                if (mutual.size >= min_intersection){
                    //calculate average
                    var sum1 = 0.0
                    var sum2 = 0.0
                    var u1rates = u_p_r_map(i)
                    var u2rates = u_p_r_map(j)
                    for (k<-mutual){
                        sum1 += u1rates(k)
                        sum2 += u2rates(k)
                    }
                    var avg1 = sum1 / mutual.size
                    var avg2 = sum2 / mutual.size
                    var numerator = 0.0
                    var denominator1 =0.0
                    var denominator2 =0.0
                    for (k<-mutual){
                        numerator += (u1rates(k) - avg1) * (u2rates(k) - avg2)
                        denominator1 += (u1rates(k) - avg1) * (u1rates(k) - avg1)
                        denominator2 += (u2rates(k) - avg2) * (u2rates(k) - avg2)
                    }
                    if (!(denominator1==0 || denominator2==0))
                    {
                        var pearson = numerator / math.sqrt(denominator1 * denominator2)
                        corrMap += ((i, j) -> pearson)
                    }
                }
            }
        }
        corrMap
    }
    def imputation (cand: Iterable[(Int,Int)], u_p_list: Array[(Int, Set[Int])],  u_p_r_map: Map[Int, Map[Int, Double]]): List[(Int, Int, Double)] ={
        var imputed =List.empty[(Int, Int, Double)]
        for (tmp<-cand){
            var u1p = u_p_list(tmp._1)
            var u2p = u_p_list(tmp._2)
            var diff12 = u1p._2.diff(u2p._2).toList
            var diff21 = u2p._2.diff(u1p._2).toList
            for (i<-diff21){
                imputed = (u1p._1, i, u_p_r_map(u2p._1)(i)) :: imputed
            }
            for (i<-diff12){
                imputed = (u2p._1, i, u_p_r_map(u1p._1)(i)) :: imputed
            }
        }
        imputed
    }

    def main(args: Array[String]): Unit = {
        val starttime = System.currentTimeMillis()
        val conf = new SparkConf().setAppName("ModelCF").setMaster("local[2]")
        var sc = new SparkContext(conf)

        // total ratings input
        var total = sc.textFile(args(0))
        total = total.filter(row => !row.contains("userID"))
        val total_ratings = total.map(_.split(',')).map(row => (row(0).toInt, row(1).toInt, row(2).toDouble))
        // testing set
        var test = sc.textFile(args(1))
        test = test.filter(row => !row.contains("userID"))
        val test_ratings = test.map(_.split(',')).map(row=> (row(0).toInt, row(1).toInt, row(2).toDouble))

        // user_product training utility matrix
        var training = total_ratings.subtract(test_ratings)
        var rdd = training.map(row =>{(row._1,(row._2,row._3))}).groupByKey().sortByKey()
//        var avg_map = rdd.mapValues(x=>{x.toSeq.map(_._2).sum / x.size}).collectAsMap()
        // user_prod_rating Map
        var u_p_r_map = rdd.map(row=>{(row._1,row._2.toMap)}).collect().toMap
        // user_prod list
        var u_p_list = rdd.map(row=>{(row._1, row._2.map(_._1).toSet)}).collect()
        // compute the correlation between users (last input is the min size of common rated items to be considered)
        var corrMap = correlation(u_p_list,u_p_r_map,6)
        println("coorMapsizebefore-----"+corrMap.size)

        // imputation 1
//        var cand = corrMap.filter(_._2>0.9).keys
//
//        var imputed = imputation(cand, u_p_list, u_p_r_map)
//        var imputedrdd = sc.parallelize(imputed)
//        val training1 = training.union(imputedrdd)
//
//        var rdd1 = training1.map(row =>{(row._1,(row._2,row._3))}).groupByKey().sortByKey()
//        u_p_r_map = rdd1.map(row=>{(row._1,row._2.toMap)}).collect().toMap
//        u_p_list = rdd1.map(row=>{(row._1, row._2.map(_._1).toSet)}).collect()
//        corrMap = correlation(u_p_list,u_p_r_map,3)
//        println("coorMapsizeafter-----"+corrMap.size)
//


        var avg_map = rdd.mapValues(x=>{x.toSeq.map(_._2).sum / x.size}).collectAsMap()
        // predict result
        var predict = test_ratings.map(row=> {
            var target = (row._1, row._2)
            var targetavg = avg_map(target._1)
            var prediction = targetavg
            var otherusers = u_p_list.filter(_._2.contains(target._2)).map(_._1)
            if (otherusers.size > 0){
                for (i<-otherusers){
                    var key = (0,0)

                    var nom = 0.0
                    var den = 0.0
                    if (i<target._1){
                        key = (i, target._1)
                    }
                    else{
                        key = (target._1, i)
                    }
                    if (corrMap.contains(key)){
                        var iavg = (u_p_r_map(i).map(_._2).sum - u_p_r_map(i)(target._2)) / (u_p_r_map(i).size - 1)
                        nom += (u_p_r_map(i)(target._2)-iavg) * corrMap(key)
                        den += math.abs(corrMap(key))
                    }
                    if (!(den == 0)){
                        prediction = targetavg + nom / den
                        if (prediction>5.0){
                            prediction = prediction - targetavg / 2
                        }
                        else if (prediction<0.0){
                            prediction = prediction + targetavg / 2
                        }
                    }
                }
            }
            (row, prediction)
        })

        var diff = predict.map(row=>{
            math.abs(row._1._3 - row._2)
        })
        val RMSE = math.sqrt(diff.map(x=>{x * x}).mean())
        var range1 = diff.filter{ case (diff) => (diff)>=0 && diff<1}.count()
        var range2 = diff.filter{ case (diff) => (diff)>=1 && diff<2}.count()
        var range3 = diff.filter{ case (diff) => (diff)>=2 && diff<3}.count()
        var range4 = diff.filter{ case (diff) => (diff)>=3 && diff<4}.count()
        var range5 = diff.filter{ case (diff) => diff>=4}.count()

        // write out result
        val writer = new PrintWriter(new File(args(2)))

        var out = predict.map(row=>{(row._1._1,row._1._2, row._2)})
            .collect().sortBy(x=>(x._1,x._2))
        for (i<-out){
            writer.write(i._1.toString+", "+i._2.toString + ", " + i._3.toString)
            writer.write("\n")
        }
        println(">=0 and <1: "+range1)
        println(">=1 and <2: "+range2)
        println(">=2 and <3: "+range3)
        println(">=3 and <4: "+range4)
        println(">=4: "+range5)
        println("RMSE: "+ RMSE)
        println("Time: "+ ((System.currentTimeMillis()-starttime)/1000).toString +" sec")
        writer.close()


    }
}