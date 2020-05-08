#include "defs.h"

double findSubGraphs(graph *G,
                     edge *maxIntWtList, int maxIntWtListSize)
{

    VERT_T *S;
    LONG_T *start;
    char *visited;
    LONG_T *pSCount;
#ifdef _OPENMP
    omp_lock_t *vLock;
#endif

    LONG_T phase_num, numPhases;
    LONG_T count;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = MPI_Wtime();

    numPhases = SubGraphPathLength + 1;

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
#endif

        VERT_T *pS, *pSt;
        LONG_T pCount, pS_size;
        LONG_T v, w, search_num;
        int tid, nthreads;
        VERT_T *endV;
        WEIGHT_T *weight;
        LONG_T *numEdges, *tnumEdges, *ttnumEdges, *unumEdges;
        WEIGHT_T *tweight, *uweight;
        VERT_T *tendV, *uendV;
        graph *G_use; //*G_temp

        LONG_T j, k, vert, n, m, i, *rank_m;

#ifdef _OPENMP
        // LONG_T i;
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#else
    tid = 0;
    nthreads = 1;
#endif

        n = N;
        m = G->m;

        pS_size = n / nthreads + 1;
        pS = (VERT_T *)malloc(pS_size * sizeof(VERT_T));
        assert(pS != NULL);
        if (rank == 0)
        {
            rank_m = (LONG_T *)malloc(size * sizeof(LONG_T));
            assert(rank_m != NULL);
        }

        if (tid == 0)
        {
            S = (VERT_T *)malloc(N * sizeof(VERT_T));
            visited = (char *)calloc(N, sizeof(char));
            start = (LONG_T *)calloc((numPhases + 2), sizeof(LONG_T));
            pSCount = (LONG_T *)malloc((nthreads + 1) * sizeof(LONG_T));
            G_use = (graph *)malloc(sizeof(graph));
            tnumEdges = (LONG_T *)malloc((N + 1) * sizeof(LONG_T));
            tendV = (VERT_T *)malloc(m * sizeof(VERT_T));
            tweight = (WEIGHT_T *)malloc(m * sizeof(WEIGHT_T));
            uendV = (VERT_T *)malloc(M * sizeof(VERT_T));
            uweight = (WEIGHT_T *)malloc(M * sizeof(WEIGHT_T));
            unumEdges = (LONG_T *)malloc((N + 1) * sizeof(LONG_T));
#ifdef _OPENMP
            vLock = (omp_lock_t *)malloc(n * sizeof(omp_lock_t));
#endif
        }
        numEdges = G->numEdges;
        endV = G->endV;
        weight = G->weight;
#if defined(MASSIVE_GRAPH)
        MPI_Allreduce(numEdges, unumEdges, (N + 1), MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

#elif defined(LARGE_GRAPH)
    MPI_Allreduce(numEdges, unumEdges, (N + 1), MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

#else
    MPI_Allreduce(numEdges, unumEdges, (N + 1), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

#endif
        if (rank == 0)
        {
            ttnumEdges = (LONG_T *)malloc((N + 1) * sizeof(LONG_T));
            for (j = 0; j < n + 1; j++)
            {
                ttnumEdges[j] = unumEdges[j];
            }
        }
        MPI_Gather(&m, sizeof(LONG_T), MPI_BYTE, rank_m, sizeof(LONG_T), MPI_BYTE, 0, MPI_COMM_WORLD);
        for (i = 0; i < size; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == i)
            {
                MPI_Send(endV, m * sizeof(VERT_T), MPI_BYTE, 0, i, MPI_COMM_WORLD);
                MPI_Send(weight, m * sizeof(WEIGHT_T), MPI_BYTE, 0, i, MPI_COMM_WORLD);
                MPI_Send(numEdges, (N + 1) * sizeof(LONG_T), MPI_BYTE, 0, i, MPI_COMM_WORLD);
            }
            if (rank == 0)
            {
                free(tendV);
                free(tweight);
                free(tnumEdges);
                tendV = (VERT_T *)malloc(rank_m[i] * sizeof(VERT_T));
                tweight = (WEIGHT_T *)malloc(rank_m[i] * sizeof(WEIGHT_T));
                tnumEdges = (LONG_T *)malloc((N + 1) * sizeof(LONG_T));
                MPI_Recv(tendV, rank_m[i] * sizeof(VERT_T), MPI_BYTE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(tweight, rank_m[i] * sizeof(WEIGHT_T), MPI_BYTE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(tnumEdges, (N + 1) * sizeof(LONG_T), MPI_BYTE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (j = 0; j < N; j++)
                {
                    for (k = tnumEdges[j] - tnumEdges[j]; k < tnumEdges[j + 1] - tnumEdges[j]; k++)
                    {
                        uendV[(k + ttnumEdges[j])] = tendV[k + tnumEdges[j]];
                        uweight[(k + ttnumEdges[j])] = tweight[k + tnumEdges[j]];
                    }
                    ttnumEdges[j] = tnumEdges[j + 1] - tnumEdges[j] + ttnumEdges[j];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(uendV, M * sizeof(VERT_T), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(uweight, M * sizeof(WEIGHT_T), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        G->m = M;
        G->n = N;
        free(G->numEdges);
        free(G->weight);
        free(G->endV);
        G->numEdges = unumEdges;
        G->weight = uweight;
        G->endV = uendV;

#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
        for (i = 0; i < n; i++)
        {
            omp_init_lock(&vLock[i]);
        }
#endif

        for (search_num = 0; search_num < maxIntWtListSize; search_num++)
        {

#ifdef _OPENMP
#pragma omp barrier
#endif
            /* Run path-limited BFS in parallel */

            if (tid == 0)
            {
                free(visited);
                visited = (char *)calloc(n, sizeof(char));
                S[0] = maxIntWtList[search_num].startVertex;
                S[1] = maxIntWtList[search_num].endVertex;
                visited[S[0]] = (char)1;
                visited[S[1]] = (char)1;
                count = 2;
                phase_num = 1;
                start[0] = 0;
                start[1] = 1;
                start[2] = 2;
            }

#ifdef _OPENMP
#pragma omp barrier
#endif

            while (phase_num <= SubGraphPathLength)
            {

                pCount = 0;

#ifdef _OPENMP
#pragma omp for
#endif
                for (vert = start[phase_num]; vert < start[phase_num + 1]; vert++)
                {

                    v = S[vert];
                    for (j = G->numEdges[v]; j < G->numEdges[v + 1]; j++)
                    {
                        w = G->endV[j];
                        if (v == w)
                            continue;
#ifdef _OPENMP
                        int myLock = omp_test_lock(&vLock[w]);
                        if (myLock)
                        {
#endif
                            if (visited[w] != (char)1)
                            {
                                visited[w] = (char)1;
                                if (pCount == pS_size)
                                {
                                    /* Resize pS */
                                    pSt = (VERT_T *)
                                        malloc(2 * pS_size * sizeof(VERT_T));
                                    memcpy(pSt, pS, pS_size * sizeof(VERT_T));
                                    free(pS);
                                    pS = pSt;
                                    pS_size = 2 * pS_size;
                                }
                                pS[pCount++] = w;
                            }
#ifdef _OPENMP
                            omp_unset_lock(&vLock[w]);
                        }
#endif
                    }
                }

#ifdef _OPENMP
#pragma omp barrier
#endif
                pSCount[tid + 1] = pCount;

#ifdef _OPENMP
#pragma omp barrier
#endif

                if (tid == 0)
                {
                    pSCount[0] = start[phase_num + 1];
                    for (k = 1; k <= nthreads; k++)
                    {
                        pSCount[k] = pSCount[k - 1] + pSCount[k];
                    }
                    start[phase_num + 2] = pSCount[nthreads];
                    count = pSCount[nthreads];
                    phase_num++;
                }

#ifdef _OPENMP
#pragma omp barrier
#endif
                for (k = pSCount[tid]; k < pSCount[tid + 1]; k++)
                {
                    S[k] = pS[k - pSCount[tid]];
                }

#ifdef _OPENMP
#pragma omp barrier
#endif
            } /* End of search */

            if (tid == 0)
            {
                fprintf(stderr, "Rank %d Search from <%ld, %ld>, number of vertices visited:"
                                " %ld\n",
                        rank, (long)S[0], (long)S[1], (long)count);
            }

        } /* End of outer loop */

        free(pS);
#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
        for (i = 0; i < n; i++)
        {
            omp_destroy_lock(&vLock[i]);
        }
#pragma omp barrier
#endif

        if (tid == 0)
        {
            free(S);
            free(start);
            free(visited);
            free(pSCount);
            if (rank == 0)
                free(ttnumEdges);
            free(tnumEdges);
            free(tweight);
            free(tendV);
#ifdef _OPENMP
            free(vLock);
#endif
        }

#ifdef _OPENMP
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime() - elapsed_time;
    return elapsed_time;
}
