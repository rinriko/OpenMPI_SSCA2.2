#include "defs.h"

/* Set this variable to zero to run the data generator 
   on one thread (for debugging purposes) */
#define PARALLEL_SDG 0

double genScalData(graphSDG *SDGdata)
{

    VERT_T *src, *dest;
    VERT_T *rank_src, *rank_dest;
    WEIGHT_T *wt;
    WEIGHT_T *rank_wt;
    LONG_T n, m, *rank_m;
    VERT_T *permV;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
#ifdef _OPENMP
    omp_lock_t *vLock;
#endif

    double elapsed_time;
    int seed;

    n = N;
    m = M;

    rank_m = (LONG_T *)malloc(size * sizeof(LONG_T));
    assert(rank_m != NULL);
    /* allocate memory for edge tuples */
    src = (VERT_T *)malloc(M * sizeof(VERT_T));
    dest = (VERT_T *)malloc(M * sizeof(VERT_T));
    assert(src != NULL);
    assert(dest != NULL);

    /* sprng seed */
    seed = 2387;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime();

#ifdef _OPENMP
#if PARALLEL_SDG
    omp_set_num_threads(omp_get_max_threads());
    // omp_set_num_threads(16);
#else
    omp_set_num_threads(1);
#endif
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        int tid, nthreads;
#ifdef DIAGNOSTIC
        double elapsed_time_part;
#endif
        int *stream;

        LONG_T i, j, u, v, step;
        DOUBLE_T av, bv, cv, dv, p, S, var;
        LONG_T tmpVal;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
#else
    nthreads = 1;
    tid = 0;
#endif

        /* Initialize RNG stream */
        stream = init_sprng(0, tid, nthreads, seed, SPRNG_DEFAULT);

#ifdef DIAGNOSTIC
        if (tid == 0)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime();
        }

#endif

        /* Start adding edges */
#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < m; i++)
        {

            u = 1;
            v = 1;
            step = n / 2;

            av = A;
            bv = B;
            cv = C;
            dv = D;

            p = sprng(stream);
            if (p < av)
            {
                /* Do nothing */
            }
            else if ((p >= av) && (p < av + bv))
            {
                v += step;
            }
            else if ((p >= av + bv) && (p < av + bv + cv))
            {
                u += step;
            }
            else
            {
                u += step;
                v += step;
            }

            for (j = 1; j < SCALE; j++)
            {
                step = step / 2;

                /* Vary a,b,c,d by up to 10% */
                var = 0.1;
                av *= 0.95 + var * sprng(stream);
                bv *= 0.95 + var * sprng(stream);
                cv *= 0.95 + var * sprng(stream);
                dv *= 0.95 + var * sprng(stream);

                S = av + bv + cv + dv;
                av = av / S;
                bv = bv / S;
                cv = cv / S;
                dv = dv / S;

                /* Choose partition */
                p = sprng(stream);
                if (p < av)
                {
                    /* Do nothing */
                }
                else if ((p >= av) && (p < av + bv))
                {
                    v += step;
                }
                else if ((p >= av + bv) && (p < av + bv + cv))
                {
                    u += step;
                }
                else
                {
                    u += step;
                    v += step;
                }
            }

            src[i] = u - 1;
            dest[i] = v - 1;
        }

#ifdef DIAGNOSTIC
        if (tid == 0)
        {

            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime() - elapsed_time_part;
            if (rank == 0)
                fprintf(stderr, "Tuple generation time: %lf seconds\n", elapsed_time_part);
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime();
        }
#endif

        /* Generate vertex ID permutations */

        if (tid == 0)
        {
            permV = (VERT_T *)malloc(N * sizeof(VERT_T));
            assert(permV != NULL);
        }

#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
#endif
        for (i = 0; i < n; i++)
        {
            permV[i] = i;
        }

#ifdef _OPENMP
        if (tid == 0)
        {
            vLock = (omp_lock_t *)malloc(n * sizeof(omp_lock_t));
            assert(vLock != NULL);
        }

#pragma omp barrier

#pragma omp for
        for (i = 0; i < n; i++)
        {
            omp_init_lock(&vLock[i]);
        }

#endif

#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < n; i++)
        {
            j = n * sprng(stream);
            if (i != j)
            {
#ifdef _OPENMP
                int l1 = omp_test_lock(&vLock[i]);
                if (l1)
                {
                    int l2 = omp_test_lock(&vLock[j]);
                    if (l2)
                    {
#endif
                        tmpVal = permV[i];
                        permV[i] = permV[j];
                        permV[j] = tmpVal;
#ifdef _OPENMP
                        omp_unset_lock(&vLock[j]);
                    }
                    omp_unset_lock(&vLock[i]);
                }
#endif
            }
        }

#ifdef _OPENMP
#pragma omp for
        for (i = 0; i < n; i++)
        {
            omp_destroy_lock(&vLock[i]);
        }

#pragma omp barrier

        if (tid == 0)
        {
            free(vLock);
        }
#endif

#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < m; i++)
        {
            src[i] = permV[src[i]];
            dest[i] = permV[dest[i]];
            if (i % size == rank)
                rank_m[rank] = i / size + 1;
        }

#ifdef DIAGNOSTIC
        if (tid == 0)
        {

            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime() - elapsed_time_part;
            if (rank == 0)
                fprintf(stderr, "Permuting vertex IDs: %lf seconds\n", elapsed_time_part);
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime();
        }
#endif

        if (tid == 0)
        {
            free(permV);
        }

        /* Generate edge weights */
        if (tid == 0)
        {
            wt = (WEIGHT_T *)malloc(M * sizeof(WEIGHT_T));
            assert(wt != NULL);
        }

#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
#endif
        for (i = 0; i < m; i++)
        {
            wt[i] = 1 + MaxIntWeight * sprng(stream);
        }

#ifdef DIAGNOSTIC
        if (tid == 0)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime() - elapsed_time_part;
            if (rank == 0)
                fprintf(stderr, "Generating edge weights: %lf seconds\n", elapsed_time_part);
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed_time_part = MPI_Wtime();
        }
#endif
        rank_src = (VERT_T *)malloc(rank_m[rank] * sizeof(VERT_T));
        assert(rank_src != NULL);
        rank_dest = (VERT_T *)malloc(rank_m[rank] * sizeof(VERT_T));
        assert(rank_dest != NULL);
        rank_wt = (WEIGHT_T *)malloc(rank_m[rank] * sizeof(WEIGHT_T));
        assert(rank_wt != NULL);
        for (i = 0; i < m; i++)
        {
            if (i % (size) == rank)
            {
                int k;
                k = (i / size);
                rank_src[k] = src[i];
                rank_wt[k] = wt[i];
                rank_dest[k] = dest[i];
            }
        }

        SDGdata->n = n;
        SDGdata->m = rank_m[rank];
        SDGdata->startVertex = rank_src;
        SDGdata->endVertex = rank_dest;
        SDGdata->weight = rank_wt;

        free_sprng(stream);
        free(src);
        free(wt);
        free(dest);
#ifdef _OPENMP
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime() - elapsed_time;
    // if (SCALE <= 5)
    // {
    //     if (rank == 0)
    //         fprintf(stderr, "Scalable Data: ");
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     for (i = 0; i < SDGdata->m; i++)
    //     {
    //         fprintf(stderr, "Rank %d = [%d, %d] - %d\n", rank, SDGdata->startVertex[i], SDGdata->endVertex[i], SDGdata->weight[i]);
    //     }
    // }
    return elapsed_time;
}
