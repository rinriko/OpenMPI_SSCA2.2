#include "defs.h"

double getStartLists(graph *G, edge **maxIntWtListPtr,
                     INT_T *maxIntWtListSizePtr, INT_T *totalMaxIntWtListSize)
{

    LONG_T *local_max, maxWeight;

    edge *maxIntWtList;
    LONG_T maxIntWtListSize;

    LONG_T *p_start, *p_end;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
    double elapsed_time;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime();

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
#endif

        LONG_T i, j, n;
        edge *pList;
        LONG_T pCount, tmpListSize;
        int tid, nthreads;
#ifdef DIAGNOSTIC
        double elapsed_time_part;
#endif

#ifdef _OPENMP
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#else
    tid = 0;
    nthreads = 1;
#endif

        n = G->n;

        /* Determine the maximum edge weight */

        if (tid == 0)
        {
            local_max = (LONG_T *)malloc(nthreads * sizeof(LONG_T));
        }

        /* Allocate memory for partial edge list on each thread */
        tmpListSize = 1000;
        pList = (edge *)malloc(tmpListSize * sizeof(edge));
        pCount = 0;

#ifdef _OPENMP
#pragma omp barrier
#endif

        local_max[tid] = -1;

#ifdef DIAGNOSTIC
        if (tid == 0)
        {

            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime();
        }
#endif

#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < n; i++)
        {
            for (j = G->numEdges[i]; j < G->numEdges[i + 1]; j++)
            {
                if (G->weight[j] > local_max[tid])
                {
                    local_max[tid] = G->weight[j];
                    pCount = 0;
                    pList[pCount].startVertex = i;
                    pList[pCount].endVertex = G->endV[j];
                    pList[pCount].w = local_max[tid];
                    pList[pCount].e = j;
                    pCount++;
                }
                else if (G->weight[j] == local_max[tid])
                {
                    pList[pCount].startVertex = i;
                    pList[pCount].endVertex = G->endV[j];
                    pList[pCount].w = local_max[tid];
                    pList[pCount].e = j;
                    pCount++;
                }
            }
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        if (tid == 0)
        {

#ifdef DIAGNOSTIC
            if (tid == 0)
            {

                MPI_Barrier(MPI_COMM_WORLD);
                elapsed_time_part = MPI_Wtime() - elapsed_time_part;
                if (rank == 0)
                    fprintf(stderr, "Max. weight computation time: %lf seconds\n", elapsed_time_part);
            }
#endif

            maxWeight = local_max[0];

            for (i = 1; i < nthreads; i++)
            {
                if (local_max[i] > maxWeight)
                    maxWeight = local_max[i];
            }
            // free(local_max);
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        if (local_max[tid] != maxWeight)
        {
            pCount = 0;
        }

        /* Merge all te partial edge lists */
        if (tid == 0)
        {
            LONG_T temp_local_max;
            MPI_Barrier(MPI_COMM_WORLD);
            temp_local_max = maxWeight;
#if defined(MASSIVE_GRAPH)
            MPI_Allreduce(&temp_local_max, &maxWeight, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
#elif defined(LARGE_GRAPH)
        MPI_Allreduce(&temp_local_max, &maxWeight, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

#else
        MPI_Allreduce(&temp_local_max, &maxWeight, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
            if (maxWeight > temp_local_max)
            {
                pCount = 0;
            }
            p_start = (LONG_T *)malloc(nthreads * sizeof(LONG_T));
            p_end = (LONG_T *)malloc(nthreads * sizeof(LONG_T));
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        p_end[tid] = pCount;
        p_start[tid] = 0;

#ifdef _OPENMP
#pragma omp barrier
#endif

        if (tid == 0)
        {
            for (i = 1; i < nthreads; i++)
            {
                p_end[i] = p_end[i - 1] + p_end[i];
                p_start[i] = p_end[i - 1];
            }

            maxIntWtListSize = p_end[nthreads - 1];
            free(*maxIntWtListPtr);
            maxIntWtList = (edge *)malloc((maxIntWtListSize) * sizeof(edge));
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        for (j = p_start[tid]; j < p_end[tid]; j++)
        {
            (maxIntWtList[j]).startVertex = pList[j - p_start[tid]].startVertex;
            (maxIntWtList[j]).endVertex = pList[j - p_start[tid]].endVertex;
            (maxIntWtList[j]).e = pList[j - p_start[tid]].e;
            (maxIntWtList[j]).w = pList[j - p_start[tid]].w;
        }

#ifdef _OPENMP
#pragma omp barrier
#endif

        free(pList);

        if (tid == 0)
        {
            free(local_max);
            free(p_start);
            free(p_end);
            *maxIntWtListPtr = maxIntWtList;
            *maxIntWtListSizePtr = maxIntWtListSize;
            LONG_T TotalmaxIntWtListSize;
#if defined(MASSIVE_GRAPH)
            MPI_Allreduce(&maxIntWtListSize, &TotalmaxIntWtListSize, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#elif defined(LARGE_GRAPH)
            MPI_Allreduce(&maxIntWtListSize, &TotalmaxIntWtListSize, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

#else
            MPI_Allreduce(&maxIntWtListSize, &TotalmaxIntWtListSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
            *totalMaxIntWtListSize = TotalmaxIntWtListSize;
        }

#ifdef _OPENMP
    }
#endif

    /* Verification */
#if 0
    maxIntWtList = *maxIntWtListPtr;
    for (int i=0; i<*maxIntWtListSizePtr; i++) {
        fprintf(stderr, "[%ld %ld %ld %ld] ", maxIntWtList[i].startVertex, 
                maxIntWtList[i].endVertex, maxIntWtList[i].e, maxIntWtList[i].w);
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime() - elapsed_time;
    // if (SCALE <= 5)
    // {
    //     if (rank == 0)
    //         fprintf(stderr, "Max weight list: \n");
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     for (i = 0; i < maxIntWtListSize; i++)
    //     {
    //         fprintf(stderr, "Rank %d\t(%d, %d) = %d\n", rank, maxIntWtList[i].startVertex, maxIntWtList[i].endVertex, maxIntWtList[i].w);
    //     }
    // }
    return elapsed_time;
}
