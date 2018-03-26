import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import java.util.Random

import scala.collection.mutable.ListBuffer


object ItemBasedCF {
    def main(args: Array[String]): Unit = {
        val starttime = System.currentTimeMillis()
        val conf = new SparkConf().setAppName("ItemCF").setMaster("local[2]")
        var sc = new SparkContext(conf)

        // total ratings input
        var total = sc.textFile(args(0))
        total = total.filter(row => !row.contains("userID"))
        val total_ratings = total.map(_.split(',')).map(row => (row(1).toInt, row(0).toInt, row(2).toDouble)) // pid,uid,rate
        // testing set
        var test = sc.textFile(args(1))
        test = test.filter(row => !row.contains("userID"))
        val test_ratings = test.map(_.split(',')).map(row=> (row(1).toInt, row(0).toInt, row(2).toDouble))  // pid,uid,rate



        // product_user training utility matrix
        var training = total_ratings.subtract(test_ratings)  //pid,uid,rate
        var rdd = training.map(row =>{(row._1,(row._2,row._3))}).groupByKey().sortByKey()
        // prod_user_rating Map
        var p_u_r_map = rdd.map(row=>{(row._1,row._2.toMap)}).collect().toMap
        // prod_user list
        var p_u_list = rdd.map(row=>{(row._1, row._2.map(_._1).toSet)}).collect()
        // each prod's avg rating
        var avg_map = rdd.mapValues(x=>{x.toSeq.map(_._2).sum / x.size}).collectAsMap()
        // compute correlation between candidate items and store in corrMap
        var corrMap = Map.empty[(Int, Int), Double]
        // set min size of intersection
        val min_intersection = 2


        // without LSH
//        val nrows = p_u_list.size
//        for (i<- 0 to nrows -1) {
//            for (j <- i + 1 to nrows - 1) {
//                var mutual = p_u_list(i)._2.intersect(p_u_list(j)._2)
//                if (mutual.size >= min_intersection) {
//                    //calculate average
//                    var sum1 = 0.0
//                    var sum2 = 0.0
//                    var u1rates = p_u_r_map(i)
//                    var u2rates = p_u_r_map(j)
//                    for (k <- mutual) {
//                        sum1 += u1rates(k)
//                        sum2 += u2rates(k)
//                    }
//                    var avg1 = sum1 / mutual.size
//                    var avg2 = sum2 / mutual.size
//                    var numerator = 0.0
//                    var denominator1 = 0.0
//                    var denominator2 = 0.0
//                    for (k <- mutual) {
//                        numerator += (u1rates(k) - avg1) * (u2rates(k) - avg2)
//                        denominator1 += (u1rates(k) - avg1) * (u1rates(k) - avg1)
//                        denominator2 += (u2rates(k) - avg2) * (u2rates(k) - avg2)
//                    }
//                    if (!(denominator1 == 0 || denominator2 == 0)) {
//                        var pearson = numerator / math.sqrt(denominator1 * denominator2)
//                        if (i < j) {
//                            corrMap += ((i, j) -> pearson)
//                        }
//                        else {
//                            corrMap += ((j, i) -> pearson)
//                        }
//                    }
//                }
//            }
//        }

        // with LSH

        // use LSH to find candidates
        var data = total_ratings.map(x=>(x._1,x._2)) //(pid, uid)
        var indexed = data.groupByKey().persist() // (pid, Iterable(uid))
        val nhash = 186
        val rows = 3
        val u = data.map(_._2).distinct().count() // 3374
        val r = new Random(5)
        var a = ListBuffer(1)
        while (a.size < nhash){
            var tmp = r.nextInt(u.toInt)
            if (!(tmp % 2 == 0 || tmp % 7 == 0 || tmp % 241 == 0)){
                a += tmp
            }
        }
        var b = Seq.fill(nhash)(r.nextInt(u.toInt))
        // minhash
        val sig_matrix = indexed.map(x=>{
            var signature = ListBuffer.empty[Long]
            for (tmp<- 0 to (nhash - 1)) {
                var minhash = u
                var tmpa = a(tmp)
                var tmpb = b(tmp)
                for (i<-x._2){
                    var hashed = (tmpa * i + tmpb) % u
                    if (hashed < minhash) {
                        minhash = hashed
                    }
                }
                signature += minhash
            }
            (x._1, signature)
        })
        // LSH algorithm to get candidates
        val candidates = sig_matrix.flatMap(row=>{
            row._2.grouped(rows).zipWithIndex.map(x=>{
                ((x._2,x._1),row._1)
            }).toList
        }).groupByKey().filter(_._2.size > 1).map(_._2.toSet).distinct().flatMap(_.subsets(2)).distinct().collect()    // (pid, pid)
        // filter with characteristic matrix
        val p_u_map = indexed.collectAsMap()
        var result = candidates.map(row=>{
            var cand = row.toList
            var x = p_u_map(cand(0)).toSet
            var y = p_u_map(cand(1)).toSet
            var j = x.intersect(y).size * 1.0 / x.union(y).size
            (row, j)
        }).filter(_._2 >= 0.5).map(_._1)

        for (cand<-result) {
            var tmp = cand.toList
            var i = tmp(0)
            var j = tmp(1)
            var mutual = p_u_list(i)._2.intersect(p_u_list(j)._2)
            if (mutual.size >= min_intersection) {
                //calculate average
                var sum1 = 0.0
                var sum2 = 0.0
                var u1rates = p_u_r_map(i)
                var u2rates = p_u_r_map(j)
                for (k <- mutual) {
                    sum1 += u1rates(k)
                    sum2 += u2rates(k)
                }
                var avg1 = sum1 / mutual.size
                var avg2 = sum2 / mutual.size
                var numerator = 0.0
                var denominator1 = 0.0
                var denominator2 = 0.0
                for (k <- mutual) {
                    numerator += (u1rates(k) - avg1) * (u2rates(k) - avg2)
                    denominator1 += (u1rates(k) - avg1) * (u1rates(k) - avg1)
                    denominator2 += (u2rates(k) - avg2) * (u2rates(k) - avg2)
                }
                if (!(denominator1 == 0 || denominator2 == 0)) {
                    var pearson = numerator / math.sqrt(denominator1 * denominator2)
                    if (i < j) {
                        corrMap += ((i, j) -> pearson)
                    }
                    else {
                        corrMap += ((j, i) -> pearson)
                    }
                }
            }
        }

        // predict result
        var predict = test_ratings.map(row=> {
            var target = (row._1, row._2)
            var targetavg = avg_map(target._1)
            var prediction = targetavg
            var otherprods = p_u_list.filter(_._2.contains(target._2)).map(_._1)
            if (otherprods.size > 0){
                for (i<-otherprods){
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
                        nom += (p_u_r_map(i)(target._2)) * corrMap(key)
                        den += math.abs(corrMap(key))
                    }
                    if (!(den == 0)){
                        prediction = nom / den
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
        // write out result
        val writer = new PrintWriter(new File(args(2)))

        var out = predict.map(row=>{(row._1._2,row._1._1, row._2)})
            .collect().sortBy(x=>(x._1,x._2))
        for (i<-out){
            writer.write(i._1.toString+", "+i._2.toString + ", " + i._3.toString)
            writer.write("\n")
        }

        var diff = predict.map(row=>{
            math.abs(row._1._3 - row._2)
        })
        val RMSE = math.sqrt(diff.map(x=>{x * x}).mean())
        var range1 = diff.filter{ case (diff) => (diff)>=0 && diff<1}.count()
        var range2 = diff.filter{ case (diff) => (diff)>=1 && diff<2}.count()
        var range3 = diff.filter{ case (diff) => (diff)>=2 && diff<3}.count()
        var range4 = diff.filter{ case (diff) => (diff)>=3 && diff<4}.count()
        var range5 = diff.filter{ case (diff) => diff>=4}.count()
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