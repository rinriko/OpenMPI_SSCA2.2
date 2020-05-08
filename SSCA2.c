#include "defs.h"

#ifdef _OPENMP
int NUM_THREADS;
#endif

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
    /* Data structure for storing generated tuples in the 
     * Scalable Data Generation Stage -- see defs.h */
    graphSDG *SDGdata;

    /* The graph data structure -- see defs.h */
    graph *G;

    /* Kernel 2 output */
    edge *maxIntWtList;
    INT_T maxIntWtListSize, totalMaxIntWtListSize;

    /* Kernel 4 output */
    DOUBLE_T *BC;

    DOUBLE_T elapsed_time;

#ifdef _OPENMP
    if (argc != 3)
    {

        fprintf(stderr, "Usage: ./SSCA2 <No. of threads> <SCALE>\n");
        exit(-1);
    }
    NUM_THREADS = atoi(argv[1]);
    SCALE = atoi(argv[2]);
    omp_set_num_threads(NUM_THREADS);
#else
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./SSCA2 <SCALE>\n");
        exit(-1);
    }
    SCALE = atoi(argv[1]);
#endif

    /* ------------------------------------ */
    /*  Initialization -- Untimed           */
    /* ------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\nHPCS SSCA Graph Analysis Benchmark v2.2\n");
        fprintf(stderr, "Running...\n\n");
    }
    init(SCALE);
    if (rank == 0)
    {
#ifdef _OPENMP
        fprintf(stderr, "\nNo. of threads: %d\n", NUM_THREADS);
#endif
        fprintf(stderr, "SCALE: %d\n\n", SCALE);
    }
    /* -------------------------------------------- */
    /*  Scalable Data Generator -- Untimed          */
    /* -------------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
#ifndef VERIFYK4
    if (rank == 0)
    {
        fprintf(stderr, "Scalable Data Generator -- ");
        fprintf(stderr, "genScalData() beginning execution...\n");
    }
    SDGdata = (graphSDG *)malloc(sizeof(graphSDG));
    elapsed_time = genScalData(SDGdata);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tgenScalData() completed execution\n");
        fprintf(stderr,
                "\nTime taken for Scalable Data Generation is %9.6lf sec.\n\n",
                elapsed_time);
    }
#else
    if (rank == 0)
    {
        fprintf(stderr, "Generating 2D torus for Kernel 4 validation -- ");
        fprintf(stderr, "gen2DTorus() beginning execution...\n");
    }
    SDGdata = (graphSDG *)malloc(sizeof(graphSDG));
    elapsed_time = gen2DTorus(SDGdata);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tgen2DTorus() completed execution\n");
        fprintf(stderr,
                "\nTime taken for 2D torus generation is %9.6lf sec.\n\n",
                elapsed_time);
    }
#endif

    /* ------------------------------------ */
    /*  Kernel 1 - Graph Construction       */
    /* ------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    /* From the SDG data, construct the graph 'G'  */
    if (rank == 0)
    {
        fprintf(stderr, "\nKernel 1 -- computeGraph() beginning execution...\n\n");
    }
    G = (graph *)malloc(sizeof(graph));
    /* Store the SDG edge lists in a compact representation 
     * which isn't modified in subsequent Kernels */
    elapsed_time = computeGraph(G, SDGdata);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tcomputeGraph() completed execution\n");
        fprintf(stderr, "\nTime taken for Kernel 1 is %9.6lf sec.\n\n",
                elapsed_time);
    }
    free(SDGdata);

    /* ---------------------------------------------------- */
    /*  Kernel 2 - Find max edge weight                     */
    /* ---------------------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\nKernel 2 -- getStartLists() beginning execution...\n\n");
    }
    /* Initialize vars and allocate temp. memory for the edge list */
    maxIntWtListSize = 0;
    maxIntWtList = (edge *)malloc(sizeof(edge));

    elapsed_time = getStartLists(G, &maxIntWtList, &maxIntWtListSize, &totalMaxIntWtListSize);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tgetStartLists() completed execution\n\n");
        fprintf(stderr, "Max. int wt. list size is %d\n", totalMaxIntWtListSize);
        fprintf(stderr, "\nTime taken for Kernel 2 is %9.6lf sec.\n\n",
                elapsed_time);
    }

    /* ------------------------------------ */
    /*  Kernel 3 - Graph Extraction         */
    /* ------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\nKernel 3 -- findSubGraphs() beginning execution...\n\n");
    }
    elapsed_time = findSubGraphs(G, maxIntWtList, maxIntWtListSize);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tfindSubGraphs() completed execution\n");
        fprintf(stderr, "\nTime taken for Kernel 3 is %9.6lf sec.\n\n",
                elapsed_time);
    }
    free(maxIntWtList);

    /* ------------------------------------------ */
    /*  Kernel 4 - Betweenness Centrality         */
    /* ------------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\nKernel 4 -- betweennessCentrality() "
                        "beginning execution...\n\n");
    }
    BC = (DOUBLE_T *)calloc(N, sizeof(DOUBLE_T));
    elapsed_time = betweennessCentrality(G, BC);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        fprintf(stderr, "\n\tbetweennessCentrality() completed execution\n");
        fprintf(stderr, "\nTime taken for Kernel 4 is %9.6f sec.\n\n",
                elapsed_time);
        fprintf(stderr, "TEPS score for Kernel 4 is %lf\n\n",
                7 * N * (1 << K4approx) / elapsed_time);
    }
    free(BC);

    /* -------------------------------------------------------------------- */

    free(G->numEdges);
    free(G->endV);
    free(G->weight);
    free(G);
    MPI_Finalize();
    return 0;
}
