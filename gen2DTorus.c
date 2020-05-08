#include "defs.h"
#define PARALLEL_SDG

/* Set this variable to zero to run the data generator 
   on one thread (for debugging purposes) */

double gen2DTorus(graphSDG *SDGdata)
{

    VERT_T *src, *dest;
    VERT_T *rank_src, *rank_dest;
    WEIGHT_T *wt;
    WEIGHT_T *rank_wt;
    LONG_T *rank_m;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */

#ifdef _OPENMP
    omp_lock_t *vLock;
#endif

    double elapsed_time;
    int seed;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime();

    /* allocate memory for edge tuples */
    src = (VERT_T *)malloc(M * sizeof(VERT_T));
    dest = (VERT_T *)malloc(M * sizeof(VERT_T));
    wt = (WEIGHT_T *)malloc(M * sizeof(WEIGHT_T));

    assert(src != NULL);
    assert(dest != NULL);
    assert(wt != NULL);
    rank_m = (LONG_T *)malloc(size * sizeof(LONG_T));
    assert(rank_m != NULL);

    /* sprng seed */
    seed = 2387;

#ifdef _OPENMP
#ifdef PARALLEL_SDG
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

        LONG_T n, m;
        LONG_T i, j, x, y;
        LONG_T x_start, x_end, offset;
        LONG_T count;

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

        n = N;
        m = M;

        if (SCALE % 2 == 0)
        {
            x = 1 << (SCALE / 2);
            y = 1 << (SCALE / 2);
        }
        else
        {
            x = 1 << ((SCALE + 1) / 2);
            y = 1 << ((SCALE - 1) / 2);
        }

        count = 0;

        x_start = (x / nthreads) * tid;
        x_end = (x / nthreads) * (tid + 1);

        if (tid == 0)
            x_start = 0;

        if (tid == nthreads - 1)
            x_end = x;

        offset = 4 * x_start * y;

        fprintf(stderr, "tid: %d, x_start: %d, x_end: %d, offset: %d\n",
                tid, x_start, x_end, offset);

        // if (tid == 0) {
        for (i = x_start; i < x_end; i++)
        {

            for (j = 0; j < y; j++)
            {

                /* go down */
                if (j > 0)
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * i + j - 1;
                }
                else
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * i + y - 1;
                }

                count++;

                /* go up */
                if (j < y - 1)
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * i + j + 1;
                }
                else
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * i;
                }

                count++;

                /* go left */
                if (i > 0)
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * (i - 1) + j;
                }
                else
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * (x - 1) + j;
                }

                count++;

                /* go right */
                if (i < x - 1)
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = y * (i + 1) + j;
                }
                else
                {
                    src[offset + count] = y * i + j;
                    dest[offset + count] = j;
                }

                count++;
            }
        }

        // }

#ifdef _OPENMP
#pragma omp barrier
#endif

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

#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
#endif
        for (i = 0; i < m; i++)
        {
            wt[i] = 1 + MaxIntWeight * sprng(stream);
            if (i % size == rank)
                rank_m[rank] = i / size + 1;
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
        rank_dest = (VERT_T *)malloc(rank_m[rank] * sizeof(VERT_T));
        rank_wt = (WEIGHT_T *)malloc(rank_m[rank] * sizeof(WEIGHT_T));
        assert(rank_src != NULL);
        assert(rank_dest != NULL);
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
    //         fprintf(stderr, "Scalable Data 2D Torus: ");
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     for (i = 0; i < SDGdata->m; i++)
    //     {
    //         fprintf(stderr, "Rank %d = [%d, %d] - %d\n", rank, SDGdata->startVertex[i], SDGdata->endVertex[i], SDGdata->weight[i]);
    //     }
    // }
    return elapsed_time;
}
