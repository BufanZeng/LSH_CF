import java.io._
import java.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ListBuffer

object JaccardLSH {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("LSH").setMaster("local[2]")
        var sc = new SparkContext(conf)
        var raw = sc.textFile(args(0))
        var header = raw.first()
        raw = raw.filter(row => row != header)
        // read data
        var data = raw.map(row => row.split(",")).map(x=>(x(1).toInt,x(0).toInt)) //(pid, uid)
        // characteristic matrix
        var indexed = data.groupByKey().persist() // (pid, Iterable(uid))

        // number of hash functions
        val nhash = 300
        // number of rows in LSH, (band will be nhash/rows)
        val rows = 3
        val u = data.map(_._2).distinct().count() // 3374
        val r = new Random(5)

        // generate a and b for hash functions (a*x +b) % u
        var a = ListBuffer(1)
        while (a.size < nhash){
            var tmp = r.nextInt(u.toInt)
            if (!(tmp % 2 == 0 || tmp % 7 == 0 || tmp % 241 == 0)){
                a += tmp
            }
        }

        var b = Seq.fill(nhash)(r.nextInt(u.toInt))

        // minhash algorithm get signature matrix
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

        // LSH algorithm to find candidates
        val candidates = sig_matrix.flatMap(row=>{
            row._2.grouped(rows).zipWithIndex.map(x=>{
                ((x._2,x._1),row._1)
            }).toList
        }).groupByKey().filter(_._2.size > 1).map(_._2.toSet).distinct().flatMap(_.subsets(2)).distinct()    // (pid, pid)

//        candidates.take(10).foreach(println)

        // filter candidates using characteristic matrix
        val p_u_map = indexed.collectAsMap()
        var result = candidates.map(row=>{
            var cand = row.toList
            val i = cand(0)
            val j = cand(1)
            var x = p_u_map(i).toSet
            var y = p_u_map(j).toSet
            var z = x.intersect(y).size * 1.0 / x.union(y).size
            if (i<j){
                (Array(i,j),z)
            }
            else {
                (Array(j,i),z)
            }
        }).filter(_._2 >= 0.5)
        //        var result = candidates
        // compare with ground trush

//        val res_count = result.count()
//        val truth_raw = sc.textFile("Data/video_small_ground_truth_jaccard.csv")
//        var truth = truth_raw.map(row => {
//            var x = row.split(",")
//            Set(x(0).toInt, x(1).toInt)
//        })
//        val num_truth = truth.count()
//        val tp = truth.intersection(result.map(_._1)).count()
//        println("TP----"+tp)
////        println("candcount_beforeJ----"+candidates.count())
////        println("numtruth----"+num_truth)
////        println("candcount_afterJ----"+res_count)
//        println("precision-----"+ tp*1.0/res_count)
//        println("recall-----"+ tp*1.0/num_truth)

        // write result to file
        val writer = new PrintWriter(new File(args(1)))

        var out = result
            .collect().sortBy(x=>(x._1(0),x._1(1)))

        for (i<-out){
            writer.write(i._1(0).toString+", "+i._1(1).toString + ", " + i._2)
            writer.write("\n")
        }
        writer.close()
    }
}
