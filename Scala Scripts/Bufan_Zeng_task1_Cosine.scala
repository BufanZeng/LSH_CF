package com.soundcloud.lsh
import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object CosineLSH {

    def main(args: Array[String]) {

        // init spark context
        val numPartitions = 8
        val input = "Data/video_small_num.csv"
        val conf = new SparkConf()
            .setAppName("LSH-Cosine")
            .setMaster("local[2]")
        val storageLevel = StorageLevel.MEMORY_AND_DISK
        val sc = new SparkContext(conf)

        // read in an example data set of word embeddings
        var raw = sc.textFile(input, numPartitions).filter(row => !row.contains("userID")).map(_.split(','))
            .map(row=> (row(1).toInt, (row(0).toInt, 1.0)))

        val data = raw.groupByKey().sortByKey()
        val ncols = raw.map(_._2._1).max() +1

        val rows = data.map {
            row =>
            IndexedRow(row._1, Vectors.sparse(ncols,row._2.toSeq)
                .toDense
            )
        }
        val matrix = new IndexedRowMatrix(rows)

        val lsh = new Lsh(
            minCosineSimilarity = 0.5,
            dimensions = 20,
            numNeighbours = 150,
            numPermutations = 18,
            partitions = numPartitions,
            storageLevel = storageLevel
        )

        val similarityMatrix = lsh.join(matrix)

        // remap both ids back to words
        val p_p_sim = similarityMatrix.entries.keyBy(_.i).keyBy(_._2.j).map(line => (Set(line._2._2.i.toInt, line._2._2.j.toInt), line._2._2.value))

        val result = p_p_sim.map(_._1)
        val res_count = result.count()
        val truth_raw = sc.textFile("Data/video_small_ground_truth_cosine.csv")
        var truth = truth_raw.map(row => {
            var x = row.split(",")
            Set(x(0).toInt, x(1).toInt)
        })

        val num_truth = truth.count()
        val tp = truth.intersection(result).count()

        var out = p_p_sim.map(row=>{(row._1.toArray, row._2)})
            .collect()
            .sortBy(x=>(x._1(0),x._1(1)))
        sc.stop()
        val writer = new PrintWriter(new File("./Bufan_Zeng_SimilarProducs_Cosine.txt"))
        for (i<-out){
            writer.write(i._1(0).toString+", "+i._1(1).toString + ", " + i._2)
            writer.write("\n")
        }
        writer.close()
        println("TP----"+tp)
        println("precision-----"+ tp*1.0/res_count)
        println("recall-----"+ tp*1.0/num_truth)

    }
}