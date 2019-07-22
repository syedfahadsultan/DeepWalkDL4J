package embedding;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;
import org.deeplearning4j.graph.iterator.parallel.GraphWalkIteratorProvider;
import org.deeplearning4j.graph.api.Vertex;
import org.neo4j.logging.Log;
import java.util.Iterator;
import org.deeplearning4j.graph.api.IVertexSequence;

import java.util.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.lang.String;
import java.lang.Integer;

public class MyRandomWalkGraphIteratorProvider<V> implements GraphWalkIteratorProvider<V> {

    private IGraph<V, ?> graph;
    private int walkLength;
    private Random rng;
    private NoEdgeHandling mode;
    private int numberOfWalks;
    private Log log;

    public MyRandomWalkGraphIteratorProvider(IGraph<V, ?> graph, int walkLength, long seed, NoEdgeHandling mode, int numberOfWalks, Log log) {
        this.graph = graph;
        this.walkLength = walkLength;
        this.rng = new Random(seed);
        this.mode = mode;
        this.numberOfWalks = numberOfWalks;
        this.log = log;
    }


    // @Override
    // public List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators) {
    //     int nVertices = graph.numVertices();
    //     numIterators = numberOfWalks;

    //     int verticesPerIter = nVertices / numIterators;

    //     List<GraphWalkIterator<V>> list = new ArrayList<>(numIterators);
    //     int last = 0;
    //     for (int i = 0; i < numIterators; i++) {
    //         int from = last;
    //         int to = Math.min(nVertices, from + verticesPerIter);
    //         if (i == numIterators - 1)
    //             to = nVertices;

    //         GraphWalkIterator<V> iter = new MyRandomWalkIterator<>(graph, walkLength, rng.nextLong(), mode, from, to);
    //         list.add(iter);
    //         last = to;
    //     }
    //     return list;
    // }

    @Override
    public List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators){
        List<GraphWalkIterator<V>> list = new ArrayList<>(numIterators);
        try{
            int nVertices = graph.numVertices();
            numIterators = this.numberOfWalks;
            int last = 0;
            
            // for (int v = 0; v < nVertices; v++) {
            for(int i=0; i < this.numberOfWalks; i++){

                GraphWalkIterator<V> iter = new RandomWalkIterator<>(this.graph, this.walkLength, this.rng.nextLong(), this.mode);

                list.add(iter);

            }
            // }
        }
        catch(Exception e){
            StringWriter writer = new StringWriter();
            PrintWriter printWriter = new PrintWriter( writer );
            e.printStackTrace( printWriter );
            printWriter.flush();

            String stackTrace = writer.toString();
            log.info(e.toString());
            log.info(e.getMessage());
            log.info(stackTrace);
        }
        return list;

    }


}
