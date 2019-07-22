package embedding;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.ResourceIterable;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;
import org.neo4j.graphdb.Relationship;

import java.util.*;
import java.io.*;
import java.lang.InterruptedException;
import java.lang.Long;

import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.index.Index;
import org.neo4j.graphdb.index.IndexManager;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.graphdb.Transaction;
import org.neo4j.internal.kernel.api.exceptions.KernelException;
import org.neo4j.graphdb.factory.GraphDatabaseSettings;

import java.nio.file.Paths;

public class DeepWalkBB {

    @Context
    public GraphDatabaseService api;

    @Context
    public Log log;


    @Procedure(name="embedding.deepwalkbb", mode=Mode.WRITE)
    @Description("Calls deepwalk from bash and writes back embeddings as node properties")
    public void deepWalk(){

        log.info("Deepwalk - BB");
    }
}

