/*
 * ================================================================
 *        THREAD POOL BASED TASK PROCESSING SYSTEM
 * ================================================================
 *
 * Operating Systems Project
 *
 * A fixed set of persistent POSIX worker threads is created once.
 * Tasks are submitted into a bounded, mutex-protected circular
 * queue; idle workers sleep on a condition variable, wake up when
 * work arrives, execute the task (real computation) and report the
 * result.  Threads are REUSED for many tasks - no thread is ever
 * created per task.
 *
 * Course Outcome mapping
 * ----------------------
 * CO-6 : Concurrency and Synchronization (PRIMARY)
 *        POSIX threads, race conditions on shared data, mutual
 *        exclusion with mutexes, condition variables for thread
 *        coordination (producer/consumer, bounded buffer),
 *        deadlock avoidance (single lock, consistent ordering).
 * CO-4 : Memory Management
 *        Dynamic allocation (calloc/free), all worker threads share
 *        ONE address space, no leaks on reset/exit.
 * CO-1 : OS as a Service Layer
 *        pthread_create()/pthread_join() are user-space wrappers
 *        over kernel services (clone(), futex()); the program is a
 *        user-space application using kernel threads.
 * CO-2 : Processes and Process Control
 *        Threads vs processes: thread creation, execution,
 *        synchronisation (join) and termination lifecycle.
 *
 * Compile:
 *   gcc -std=c11 -Wall -Wextra -O2 -pthread thread_pool_system.c \
 *       -o thread_pool_system
 *
 * Run:
 *   ./thread_pool_system
 *
 * Platform: Linux / Ubuntu / WSL, GCC.
 * ================================================================
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>
#include <pthread.h>

/* ================================================================
   CONSTANTS AND LIMITS
   ================================================================ */

#define MAX_WORKERS          16
#define MAX_TASKS            1000
#define MAX_QUEUE_CAPACITY   1000
#define MAX_LIST_ELEMENTS    100
#define RESULT_LEN           160
#define PARAM_LEN            80

/* Parameter limits chosen so that every result fits in 64 bits   */
#define SUM_MAX_N            100000000LL          /* 1e8           */
#define FACT_MAX_N           20LL                 /* 20! fits u64  */
#define PRIME_MAX_N          1000000000000LL      /* 1e12          */
#define LIST_VALUE_LIMIT     1000000000LL         /* |value| <= 1e9*/
#define WORKLOAD_MAX_ITER    50000000LL           /* 5e7           */

/* ================================================================
   DATA STRUCTURES
   ================================================================ */

typedef enum
{
    TASK_SUM = 1,
    TASK_FACTORIAL,
    TASK_PRIME,
    TASK_LIST_STATS,
    TASK_WORKLOAD
} TaskType;

typedef enum
{
    STATUS_PENDING = 0,   /* waiting in the queue            */
    STATUS_RUNNING,       /* a worker is executing it        */
    STATUS_COMPLETED,     /* finished successfully           */
    STATUS_FAILED         /* rejected / error during run     */
} TaskStatus;

typedef struct
{
    int        id;                         /* unique task ID (1..)     */
    TaskType   type;
    TaskStatus status;
    long long  param;                      /* N / number / iterations  */
    long long  list[MAX_LIST_ELEMENTS];    /* for LIST_STATS           */
    int        list_count;
    char       result[RESULT_LEN];         /* result or error message  */
    int        worker_id;                  /* 0 = not processed yet    */
    double     exec_ms;                    /* execution time           */
} Task;

typedef struct
{
    int  ok;                               /* 1 = success, 0 = failure */
    char text[RESULT_LEN];
} Outcome;

struct ThreadPool;

typedef struct
{
    struct ThreadPool *pool;
    int                id;                 /* logical worker ID 1..N   */
} WorkerArg;

typedef struct ThreadPool
{
    int  configured;
    int  num_workers;
    int  queue_capacity;

    /* Worker threads (persistent) */
    pthread_t *threads;
    WorkerArg *args;
    int        threads_created;

    /* Bounded circular queue holding indexes into tasks[] */
    int *queue;
    int  head;
    int  tail;
    int  count;

    /* Task table: records every task ever submitted */
    Task *tasks;
    int   total_tasks;

    /* Counters (all protected by lock) */
    int completed;
    int failed;
    int running_now;

    /* Per-worker statistics */
    int    *worker_tasks;
    double *worker_busy_ms;

    /* Control flags (protected by lock) */
    int processing_enabled;   /* workers may take tasks only if 1     */
    int shutdown;             /* tells workers to exit                */
    int trace;                /* print per-task worker messages       */

    /*
     * SYNCHRONISATION
     *  lock      : one mutex guards ALL shared pool state, so there
     *              is exactly one lock -> no lock-ordering deadlock.
     *  not_empty : workers wait here while there is nothing to do.
     *  not_full  : producers wait here when the queue is full.
     *  all_done  : main thread waits here until every task is done.
     */
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
    pthread_cond_t  all_done;
    int             sync_ready;
} ThreadPool;

typedef struct
{
    int configured;
    int workers;
    int capacity;
    int submitted;
    int completed;
    int failed;
    int pending;      /* submitted - completed - failed */
    int queued;       /* waiting in queue               */
    int running;      /* currently executing            */
    int processing;
} PoolSnapshot;

typedef enum
{
    SUBMIT_OK = 0,
    SUBMIT_QUEUE_FULL,
    SUBMIT_TABLE_FULL,
    SUBMIT_SHUTDOWN
} SubmitResult;

/* ================================================================
   GLOBALS
   ================================================================ */

static ThreadPool g_pool;                  /* the one thread pool      */

/* Serialises console output coming from several threads at once */
static pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;

/* ================================================================
   UTILITY FUNCTIONS
   ================================================================ */

static void print_line(void)
{
    printf("------------------------------------------------------------------------------------\n");
}

static void print_header(const char *title)
{
    printf("\n====================================================================================\n");
    printf(" %s\n", title);
    printf("====================================================================================\n");
}

/* Thread-safe printf used by workers and by the producer */
static void log_event(const char *fmt, ...)
{
    va_list ap;

    pthread_mutex_lock(&print_lock);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    pthread_mutex_unlock(&print_lock);
}

/* Append formatted text to a buffer without overflowing it */
static void appendf(char *buf, size_t size, const char *fmt, ...)
{
    size_t  len = strlen(buf);
    va_list ap;

    if (len >= size - 1)
        return;

    va_start(ap, fmt);
    vsnprintf(buf + len, size - len, fmt, ap);
    va_end(ap);
}

/* Monotonic clock in milliseconds */
static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static const char *task_type_name(TaskType type)
{
    switch (type)
    {
    case TASK_SUM:        return "SUM";
    case TASK_FACTORIAL:  return "FACTORIAL";
    case TASK_PRIME:      return "PRIME";
    case TASK_LIST_STATS: return "LIST_STATS";
    case TASK_WORKLOAD:   return "WORKLOAD";
    }
    return "UNKNOWN";
}

static const char *status_name(TaskStatus status)
{
    switch (status)
    {
    case STATUS_PENDING:   return "PENDING";
    case STATUS_RUNNING:   return "RUNNING";
    case STATUS_COMPLETED: return "COMPLETED";
    case STATUS_FAILED:    return "FAILED";
    }
    return "UNKNOWN";
}

/* Short human-readable description of a task's parameters */
static void describe_params(const Task *t, char *buf, size_t size)
{
    buf[0] = '\0';

    switch (t->type)
    {
    case TASK_SUM:
        appendf(buf, size, "N=%lld", t->param);
        break;
    case TASK_FACTORIAL:
        appendf(buf, size, "n=%lld", t->param);
        break;
    case TASK_PRIME:
        appendf(buf, size, "n=%lld", t->param);
        break;
    case TASK_WORKLOAD:
        appendf(buf, size, "iterations=%lld", t->param);
        break;
    case TASK_LIST_STATS:
        appendf(buf, size, "[");
        for (int i = 0; i < t->list_count; i++)
        {
            if (i >= 3)
            {
                appendf(buf, size, ", ...");
                break;
            }
            appendf(buf, size, i == 0 ? "%lld" : ", %lld", t->list[i]);
        }
        appendf(buf, size, "] (%d values)", t->list_count);
        break;
    }
}

/* ================================================================
   INPUT HELPERS (no scanf / gets: everything goes through fgets)
   ================================================================ */

static void pool_teardown(ThreadPool *p);   /* forward declaration */

/* Called when stdin is closed: clean up and leave gracefully */
static void handle_eof(void)
{
    printf("\n\nInput closed (EOF). Shutting down thread pool...\n");
    pool_teardown(&g_pool);
    exit(0);
}

/* Read one line safely; discards any characters that do not fit */
static int read_line(char *buf, size_t size)
{
    if (fgets(buf, (int)size, stdin) == NULL)
        return 0;

    size_t len = strlen(buf);

    if (len > 0 && buf[len - 1] == '\n')
    {
        buf[len - 1] = '\0';
    }
    else
    {
        int c;
        while ((c = getchar()) != '\n' && c != EOF)
            ;
    }
    return 1;
}

/* Parse a complete line as one whole number */
static int parse_ll(const char *text, long long *out)
{
    char *end;

    while (isspace((unsigned char)*text))
        text++;

    errno = 0;
    long long v = strtoll(text, &end, 10);

    if (end == text || errno == ERANGE)
        return 0;

    while (isspace((unsigned char)*end))
        end++;

    if (*end != '\0')
        return 0;

    *out = v;
    return 1;
}

/* Keep asking until the user enters a whole number in [lo, hi] */
static long long prompt_ll(const char *prompt, long long lo, long long hi)
{
    char      line[128];
    long long value;

    for (;;)
    {
        printf("%s [%lld - %lld]: ", prompt, lo, hi);
        fflush(stdout);

        if (!read_line(line, sizeof(line)))
            handle_eof();

        if (!parse_ll(line, &value))
        {
            printf("  Invalid input. Please enter a whole number.\n");
            continue;
        }

        if (value < lo || value > hi)
        {
            printf("  Out of range. Allowed range is %lld to %lld.\n", lo, hi);
            continue;
        }

        return value;
    }
}

static int prompt_yes_no(const char *question)
{
    char line[64];

    for (;;)
    {
        printf("%s (y/n): ", question);
        fflush(stdout);

        if (!read_line(line, sizeof(line)))
            handle_eof();

        if (line[0] == 'y' || line[0] == 'Y')
            return 1;
        if (line[0] == 'n' || line[0] == 'N')
            return 0;

        printf("  Please answer y or n.\n");
    }
}

static void pause_screen(void)
{
    char line[64];

    printf("\nPress ENTER to continue...");
    fflush(stdout);

    if (!read_line(line, sizeof(line)))
        handle_eof();
}

/* Read exactly 'count' integers (may span several lines) */
static void read_int_list(long long *list, int count)
{
    char line[4096];

    for (;;)
    {
        int         got = 0;
        const char *error = NULL;

        printf("Enter %d integers (each between %lld and %lld),\n"
               "separated by spaces (you may continue on the next line):\n",
               count, -LIST_VALUE_LIMIT, LIST_VALUE_LIMIT);

        while (got < count && error == NULL)
        {
            printf("  > ");
            fflush(stdout);

            if (!read_line(line, sizeof(line)))
                handle_eof();

            const char *p = line;

            for (;;)
            {
                while (isspace((unsigned char)*p))
                    p++;
                if (*p == '\0')
                    break;

                char *end;
                errno = 0;
                long long v = strtoll(p, &end, 10);

                if (end == p || errno == ERANGE ||
                    (*end != '\0' && !isspace((unsigned char)*end)))
                {
                    error = "Invalid value found (only whole numbers allowed).";
                    break;
                }
                if (v < -LIST_VALUE_LIMIT || v > LIST_VALUE_LIMIT)
                {
                    error = "A value is outside the allowed range.";
                    break;
                }
                if (got >= count)
                {
                    error = "More values entered than requested.";
                    break;
                }

                list[got++] = v;
                p = end;
            }
        }

        if (error == NULL)
            return;

        printf("  Error: %s Please enter the whole list again.\n", error);
    }
}

/* ================================================================
   TASK VALIDATION AND EXECUTION
   ================================================================ */

/* Returns 1 if the task parameters are valid, otherwise 0 + message */
static int validate_task(const Task *t, char *err, size_t size)
{
    switch (t->type)
    {
    case TASK_SUM:
        if (t->param < 1 || t->param > SUM_MAX_N)
        {
            snprintf(err, size, "SUM needs 1 <= N <= %lld", SUM_MAX_N);
            return 0;
        }
        return 1;

    case TASK_FACTORIAL:
        if (t->param < 0 || t->param > FACT_MAX_N)
        {
            snprintf(err, size, "FACTORIAL needs 0 <= n <= %lld", FACT_MAX_N);
            return 0;
        }
        return 1;

    case TASK_PRIME:
        if (t->param < 0 || t->param > PRIME_MAX_N)
        {
            snprintf(err, size, "PRIME needs 0 <= n <= %lld", PRIME_MAX_N);
            return 0;
        }
        return 1;

    case TASK_LIST_STATS:
        if (t->list_count < 1 || t->list_count > MAX_LIST_ELEMENTS)
        {
            snprintf(err, size, "LIST needs 1..%d values", MAX_LIST_ELEMENTS);
            return 0;
        }
        for (int i = 0; i < t->list_count; i++)
        {
            if (t->list[i] < -LIST_VALUE_LIMIT || t->list[i] > LIST_VALUE_LIMIT)
            {
                snprintf(err, size, "LIST value out of range");
                return 0;
            }
        }
        return 1;

    case TASK_WORKLOAD:
        if (t->param < 1 || t->param > WORKLOAD_MAX_ITER)
        {
            snprintf(err, size, "WORKLOAD needs 1 <= iterations <= %lld",
                     WORKLOAD_MAX_ITER);
            return 0;
        }
        return 1;
    }

    snprintf(err, size, "unknown task type");
    return 0;
}

/*
 * Performs the real computation for one task.
 * It works only on its own private copy of the task and a private
 * Outcome, so it needs no locking.
 */
static void execute_task(const Task *t, Outcome *out)
{
    char err[PARAM_LEN];

    out->ok = 0;
    out->text[0] = '\0';

    /* Defensive re-validation: never compute with bad parameters */
    if (!validate_task(t, err, sizeof(err)))
    {
        snprintf(out->text, sizeof(out->text), "ERROR: %s", err);
        return;
    }

    switch (t->type)
    {
    case TASK_SUM:
    {
        unsigned long long sum = 0;

        for (long long i = 1; i <= t->param; i++)
            sum += (unsigned long long)i;

        /* Cross-check with Gauss' closed form n(n+1)/2 */
        unsigned long long n = (unsigned long long)t->param;
        unsigned long long expected = (n * (n + 1ULL)) / 2ULL;

        if (sum != expected)
        {
            snprintf(out->text, sizeof(out->text),
                     "ERROR: internal check failed for SUM");
            return;
        }
        snprintf(out->text, sizeof(out->text),
                 "Sum(1..%lld) = %llu", t->param, sum);
        out->ok = 1;
        break;
    }

    case TASK_FACTORIAL:
    {
        unsigned long long f = 1;

        for (long long i = 2; i <= t->param; i++)
        {
            if (__builtin_mul_overflow(f, (unsigned long long)i, &f))
            {
                snprintf(out->text, sizeof(out->text),
                         "ERROR: overflow computing %lld!", t->param);
                return;
            }
        }
        snprintf(out->text, sizeof(out->text),
                 "%lld! = %llu", t->param, f);
        out->ok = 1;
        break;
    }

    case TASK_PRIME:
    {
        unsigned long long n = (unsigned long long)t->param;
        unsigned long long factor = 0;      /* smallest factor found */

        if (n < 2)
            factor = 1;                     /* 0 and 1 are not prime */
        else if (n < 4)
            factor = 0;                     /* 2 and 3 are prime     */
        else if (n % 2 == 0)
            factor = 2;
        else if (n % 3 == 0)
            factor = 3;
        else
        {
            /* Trial division by numbers of the form 6k +/- 1 */
            for (unsigned long long i = 5; i * i <= n; i += 6)
            {
                if (n % i == 0)       { factor = i;     break; }
                if (n % (i + 2) == 0) { factor = i + 2; break; }
            }
        }

        if (n < 2)
            snprintf(out->text, sizeof(out->text),
                     "%llu is NOT prime (primes start at 2)", n);
        else if (factor == 0)
            snprintf(out->text, sizeof(out->text), "%llu is PRIME", n);
        else
            snprintf(out->text, sizeof(out->text),
                     "%llu is NOT prime (divisible by %llu)", n, factor);

        out->ok = 1;
        break;
    }

    case TASK_LIST_STATS:
    {
        long long sum = 0;
        long long min = t->list[0];
        long long max = t->list[0];

        /* |value| <= 1e9 and <= 100 values, so sum cannot overflow */
        for (int i = 0; i < t->list_count; i++)
        {
            sum += t->list[i];
            if (t->list[i] < min) min = t->list[i];
            if (t->list[i] > max) max = t->list[i];
        }

        double avg = (double)sum / (double)t->list_count;

        snprintf(out->text, sizeof(out->text),
                 "Count=%d Sum=%lld Avg=%.2f Min=%lld Max=%lld",
                 t->list_count, sum, avg, min, max);
        out->ok = 1;
        break;
    }

    case TASK_WORKLOAD:
    {
        /* Genuine CPU-bound loop: FNV-style hash mixing */
        unsigned long long h = 1469598103934665603ULL;

        for (long long i = 0; i < t->param; i++)
        {
            h ^= (unsigned long long)i;
            h *= 1099511628211ULL;
            h = (h << 13) | (h >> 51);
        }
        snprintf(out->text, sizeof(out->text),
                 "Completed %lld iterations, checksum = 0x%016llx",
                 t->param, h);
        out->ok = 1;
        break;
    }
    }
}

/* ================================================================
   THREAD POOL CORE
   ================================================================ */

/*
 * WORKER THREAD
 *
 * Each worker loops forever:
 *   1. Lock the pool mutex.
 *   2. While there is nothing it may take (queue empty OR processing
 *      not enabled) wait on 'not_empty'.  pthread_cond_wait()
 *      atomically releases the mutex and sleeps, then re-acquires
 *      the mutex when woken.  The wait is inside a while() loop to
 *      cope with spurious wake-ups.
 *   3. Remove one task index from the queue (the critical section is
 *      kept short: only queue bookkeeping happens under the lock).
 *   4. UNLOCK, then run the task - this is what lets several workers
 *      compute truly in parallel on different CPU cores.
 *   5. Lock again, publish status/result, update counters, and wake
 *      the main thread if every submitted task is now finished.
 *
 * SCHEDULING: tasks leave the queue in FIFO order; whichever worker
 * becomes free first takes the next task (dynamic load balancing).
 */
static void *worker_main(void *arg)
{
    WorkerArg  *wa = (WorkerArg *)arg;
    ThreadPool *p  = wa->pool;
    int         wid = wa->id;

    pthread_mutex_lock(&p->lock);

    for (;;)
    {
        while (!p->shutdown && (!p->processing_enabled || p->count == 0))
            pthread_cond_wait(&p->not_empty, &p->lock);

        if (p->shutdown)
            break;

        /* ---- critical section: take one task from the queue ---- */
        int   index = p->queue[p->head];
        Task *slot  = &p->tasks[index];

        p->head = (p->head + 1) % p->queue_capacity;
        p->count--;

        slot->status    = STATUS_RUNNING;
        slot->worker_id = wid;
        p->running_now++;

        Task local = *slot;              /* private copy to work on */
        int  trace = p->trace;

        pthread_cond_signal(&p->not_full);   /* a queue slot is free */
        pthread_mutex_unlock(&p->lock);
        /* ---- end critical section ---- */

        char   params[PARAM_LEN];
        Outcome outcome;

        describe_params(&local, params, sizeof(params));

        if (trace)
            log_event("  [Worker %2d] START  task #%-4d %-10s %s\n",
                      wid, local.id, task_type_name(local.type), params);

        double start = now_ms();
        execute_task(&local, &outcome);     /* real work, no lock held */
        double elapsed = now_ms() - start;

        if (trace)
            log_event("  [Worker %2d] FINISH task #%-4d (%.2f ms) -> %s\n",
                      wid, local.id, elapsed, outcome.text);

        /* ---- critical section: publish result ---- */
        pthread_mutex_lock(&p->lock);

        slot->status = outcome.ok ? STATUS_COMPLETED : STATUS_FAILED;
        strncpy(slot->result, outcome.text, RESULT_LEN - 1);
        slot->result[RESULT_LEN - 1] = '\0';
        slot->exec_ms = elapsed;

        p->running_now--;
        if (outcome.ok)
            p->completed++;
        else
            p->failed++;

        p->worker_tasks[wid - 1]++;
        p->worker_busy_ms[wid - 1] += elapsed;

        if (p->completed + p->failed == p->total_tasks)
            pthread_cond_broadcast(&p->all_done);
    }

    pthread_mutex_unlock(&p->lock);
    return NULL;
}

/* Free every heap block owned by the pool */
static void pool_free_memory(ThreadPool *p)
{
    free(p->threads);
    free(p->args);
    free(p->queue);
    free(p->tasks);
    free(p->worker_tasks);
    free(p->worker_busy_ms);
}

/*
 * Stop all workers, join them, destroy synchronisation objects and
 * free memory.  Safe to call on a partially built or empty pool.
 */
static void pool_teardown(ThreadPool *p)
{
    if (p->sync_ready)
    {
        pthread_mutex_lock(&p->lock);
        p->shutdown = 1;
        p->processing_enabled = 0;
        pthread_cond_broadcast(&p->not_empty);   /* wake sleeping workers */
        pthread_cond_broadcast(&p->not_full);
        pthread_mutex_unlock(&p->lock);

        for (int i = 0; i < p->threads_created; i++)
        {
            int rc = pthread_join(p->threads[i], NULL);
            if (rc != 0)
                fprintf(stderr, "pthread_join(worker %d) failed: %s\n",
                        i + 1, strerror(rc));
        }

        pthread_cond_destroy(&p->all_done);
        pthread_cond_destroy(&p->not_full);
        pthread_cond_destroy(&p->not_empty);
        pthread_mutex_destroy(&p->lock);
    }

    pool_free_memory(p);
    memset(p, 0, sizeof(*p));
}

/*
 * Create the pool: allocate memory, initialise the mutex and
 * condition variables, then start 'workers' persistent threads.
 * Returns 0 on success, -1 on failure (everything is cleaned up).
 */
static int pool_init(ThreadPool *p, int workers, int capacity)
{
    int rc;

    memset(p, 0, sizeof(*p));
    p->num_workers    = workers;
    p->queue_capacity = capacity;

    p->threads        = calloc((size_t)workers, sizeof(pthread_t));
    p->args           = calloc((size_t)workers, sizeof(WorkerArg));
    p->queue          = calloc((size_t)capacity, sizeof(int));
    p->tasks          = calloc(MAX_TASKS, sizeof(Task));
    p->worker_tasks   = calloc((size_t)workers, sizeof(int));
    p->worker_busy_ms = calloc((size_t)workers, sizeof(double));

    if (!p->threads || !p->args || !p->queue || !p->tasks ||
        !p->worker_tasks || !p->worker_busy_ms)
    {
        fprintf(stderr, "Error: out of memory while creating the pool.\n");
        pool_free_memory(p);
        memset(p, 0, sizeof(*p));
        return -1;
    }

    if ((rc = pthread_mutex_init(&p->lock, NULL)) != 0)
    {
        fprintf(stderr, "Error: pthread_mutex_init: %s\n", strerror(rc));
        goto fail_memory;
    }
    if ((rc = pthread_cond_init(&p->not_empty, NULL)) != 0)
    {
        fprintf(stderr, "Error: pthread_cond_init(not_empty): %s\n", strerror(rc));
        goto fail_mutex;
    }
    if ((rc = pthread_cond_init(&p->not_full, NULL)) != 0)
    {
        fprintf(stderr, "Error: pthread_cond_init(not_full): %s\n", strerror(rc));
        goto fail_not_empty;
    }
    if ((rc = pthread_cond_init(&p->all_done, NULL)) != 0)
    {
        fprintf(stderr, "Error: pthread_cond_init(all_done): %s\n", strerror(rc));
        goto fail_not_full;
    }
    p->sync_ready = 1;

    /* Start the persistent worker threads */
    for (int i = 0; i < workers; i++)
    {
        p->args[i].pool = p;
        p->args[i].id   = i + 1;

        rc = pthread_create(&p->threads[i], NULL, worker_main, &p->args[i]);
        if (rc != 0)
        {
            fprintf(stderr,
                    "Error: pthread_create failed for worker %d: %s\n",
                    i + 1, strerror(rc));
            fprintf(stderr,
                    "Stopping the %d worker(s) already created...\n", i);

            p->threads_created = i;      /* only join what exists */
            pool_teardown(p);
            return -1;
        }
        p->threads_created = i + 1;
    }

    p->configured = 1;
    return 0;

fail_not_full:
    pthread_cond_destroy(&p->not_full);
fail_not_empty:
    pthread_cond_destroy(&p->not_empty);
fail_mutex:
    pthread_mutex_destroy(&p->lock);
fail_memory:
    pool_free_memory(p);
    memset(p, 0, sizeof(*p));
    return -1;
}

/* Copy counters while holding the lock (consistent view) */
static PoolSnapshot pool_snapshot(ThreadPool *p)
{
    PoolSnapshot s;

    memset(&s, 0, sizeof(s));

    if (!p->configured)
        return s;

    pthread_mutex_lock(&p->lock);
    s.configured = 1;
    s.workers    = p->num_workers;
    s.capacity   = p->queue_capacity;
    s.submitted  = p->total_tasks;
    s.completed  = p->completed;
    s.failed     = p->failed;
    s.queued     = p->count;
    s.running    = p->running_now;
    s.pending    = p->total_tasks - p->completed - p->failed;
    s.processing = p->processing_enabled;
    pthread_mutex_unlock(&p->lock);

    return s;
}

/*
 * Put a task into the bounded queue (PRODUCER side).
 *  blocking = 0 : fail immediately if the queue is full
 *  blocking = 1 : wait on 'not_full' until a worker frees a slot
 */
static SubmitResult pool_submit(ThreadPool *p, const Task *proto,
                                int blocking, int *out_id)
{
    pthread_mutex_lock(&p->lock);

    if (blocking)
    {
        while (p->count == p->queue_capacity && !p->shutdown)
            pthread_cond_wait(&p->not_full, &p->lock);
    }

    if (p->shutdown)
    {
        pthread_mutex_unlock(&p->lock);
        return SUBMIT_SHUTDOWN;
    }
    if (p->total_tasks >= MAX_TASKS)
    {
        pthread_mutex_unlock(&p->lock);
        return SUBMIT_TABLE_FULL;
    }
    if (p->count == p->queue_capacity)
    {
        pthread_mutex_unlock(&p->lock);
        return SUBMIT_QUEUE_FULL;
    }

    int   index = p->total_tasks;
    Task *slot  = &p->tasks[index];

    *slot = *proto;
    slot->id        = index + 1;
    slot->status    = STATUS_PENDING;
    slot->worker_id = 0;
    slot->exec_ms   = 0.0;
    strcpy(slot->result, "-");

    p->queue[p->tail] = index;
    p->tail = (p->tail + 1) % p->queue_capacity;
    p->count++;
    p->total_tasks++;

    if (out_id)
        *out_id = slot->id;

    pthread_cond_signal(&p->not_empty);      /* wake one idle worker */
    pthread_mutex_unlock(&p->lock);

    return SUBMIT_OK;
}

/* Allow workers to start taking tasks */
static void pool_begin_processing(ThreadPool *p, int trace)
{
    pthread_mutex_lock(&p->lock);
    p->trace = trace;
    p->processing_enabled = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_mutex_unlock(&p->lock);
}

/* Block until every submitted task has finished */
static void pool_wait_idle(ThreadPool *p)
{
    pthread_mutex_lock(&p->lock);
    while (p->completed + p->failed < p->total_tasks)
        pthread_cond_wait(&p->all_done, &p->lock);
    pthread_mutex_unlock(&p->lock);
}

/* Send workers back to sleep (they stay alive for reuse) */
static void pool_pause_processing(ThreadPool *p)
{
    pthread_mutex_lock(&p->lock);
    p->processing_enabled = 0;
    pthread_mutex_unlock(&p->lock);
}

/* ================================================================
   DISPLAY HELPERS
   ================================================================ */

static void print_pool_summary(const PoolSnapshot *s)
{
    print_line();
    printf("Configured worker threads : %d\n", s->workers);
    printf("Queue capacity            : %d\n", s->capacity);
    printf("Tasks submitted           : %d\n", s->submitted);
    printf("Tasks completed           : %d\n", s->completed);
    printf("Tasks pending             : %d  (queued: %d, running: %d)\n",
           s->pending, s->queued, s->running);
    printf("Tasks failed              : %d\n", s->failed);
    print_line();
}

static void print_worker_stats(ThreadPool *p)
{
    pthread_mutex_lock(&p->lock);

    printf("\nPer-worker statistics:\n");
    printf("  %-10s %-12s %s\n", "Worker", "Tasks Done", "Busy Time (ms)");
    for (int i = 0; i < p->num_workers; i++)
        printf("  Worker %-3d %-12d %.2f\n",
               i + 1, p->worker_tasks[i], p->worker_busy_ms[i]);

    pthread_mutex_unlock(&p->lock);
}

/* Table of every task: used by menu option 5 and the simulation */
static void print_task_table(ThreadPool *p)
{
    pthread_mutex_lock(&p->lock);

    if (p->total_tasks == 0)
    {
        printf("No tasks have been submitted yet.\n");
        pthread_mutex_unlock(&p->lock);
        return;
    }

    printf("%-5s %-11s %-34s %-10s %-7s %-10s %s\n",
           "ID", "Type", "Parameters", "Status", "Worker", "Time(ms)", "Result");
    print_line();

    for (int i = 0; i < p->total_tasks; i++)
    {
        const Task *t = &p->tasks[i];
        char params[PARAM_LEN];
        char worker[16];
        char time_text[24];

        describe_params(t, params, sizeof(params));

        if (t->worker_id > 0)
            snprintf(worker, sizeof(worker), "W%d", t->worker_id);
        else
            snprintf(worker, sizeof(worker), "-");

        if (t->status == STATUS_COMPLETED || t->status == STATUS_FAILED)
            snprintf(time_text, sizeof(time_text), "%.2f", t->exec_ms);
        else
            snprintf(time_text, sizeof(time_text), "-");

        printf("%-5d %-11s %-34s %-10s %-7s %-10s %s\n",
               t->id, task_type_name(t->type), params,
               status_name(t->status), worker, time_text, t->result);
    }

    pthread_mutex_unlock(&p->lock);
}

/* ================================================================
   FEATURE 1: CONFIGURE THREAD POOL
   ================================================================ */

static void configure_thread_pool(void)
{
    print_header("1. CONFIGURE THREAD POOL");

    if (g_pool.configured)
    {
        PoolSnapshot s = pool_snapshot(&g_pool);

        /* Never reconfigure while workers are executing tasks */
        if (s.processing || s.running > 0)
        {
            printf("Cannot reconfigure: worker threads are currently active.\n");
            printf("Wait until processing finishes.\n");
            return;
        }

        printf("A thread pool is already configured:\n");
        print_pool_summary(&s);

        if (!prompt_yes_no("Reconfiguring stops the current workers and DISCARDS all tasks. Continue?"))
        {
            printf("Configuration unchanged.\n");
            return;
        }

        pool_teardown(&g_pool);
        printf("Previous thread pool shut down.\n\n");
    }

    int workers  = (int)prompt_ll("Number of worker threads", 1, MAX_WORKERS);
    int capacity = (int)prompt_ll("Maximum task queue capacity", 1, MAX_QUEUE_CAPACITY);

    printf("\nCreating %d persistent worker thread(s)...\n", workers);

    if (pool_init(&g_pool, workers, capacity) != 0)
    {
        printf("Thread pool creation FAILED. The pool is not configured.\n");
        return;
    }

    printf("Thread pool created. Workers are idle and waiting for tasks.\n");
    PoolSnapshot s = pool_snapshot(&g_pool);
    print_pool_summary(&s);
}

/* ================================================================
   FEATURE 2: SUBMIT TASKS
   ================================================================ */

/* Ask the user for a task type and its parameters */
static void build_task_interactive(Task *t)
{
    memset(t, 0, sizeof(*t));

    printf("\nTask types:\n");
    printf("  1. Sum of integers 1..N              (1 <= N <= %lld)\n", SUM_MAX_N);
    printf("  2. Factorial of n                    (0 <= n <= %lld)\n", FACT_MAX_N);
    printf("  3. Prime check of n                  (0 <= n <= %lld)\n", PRIME_MAX_N);
    printf("  4. Sum and average of a list         (1..%d integers)\n", MAX_LIST_ELEMENTS);
    printf("  5. Simulated workload (iterations)   (1..%lld)\n", WORKLOAD_MAX_ITER);

    t->type = (TaskType)prompt_ll("Select task type", 1, 5);

    switch (t->type)
    {
    case TASK_SUM:
        t->param = prompt_ll("Enter N", 1, SUM_MAX_N);
        break;
    case TASK_FACTORIAL:
        t->param = prompt_ll("Enter n", 0, FACT_MAX_N);
        break;
    case TASK_PRIME:
        t->param = prompt_ll("Enter the number to test", 0, PRIME_MAX_N);
        break;
    case TASK_LIST_STATS:
        t->list_count = (int)prompt_ll("How many integers", 1, MAX_LIST_ELEMENTS);
        read_int_list(t->list, t->list_count);
        break;
    case TASK_WORKLOAD:
        t->param = prompt_ll("Number of iterations", 1, WORKLOAD_MAX_ITER);
        break;
    }
}

static void submit_tasks(void)
{
    print_header("2. SUBMIT TASKS");

    if (!g_pool.configured)
    {
        printf("The thread pool is not configured. Use option 1 first.\n");
        return;
    }

    PoolSnapshot s = pool_snapshot(&g_pool);
    int free_slots = s.capacity - s.queued;
    int table_left = MAX_TASKS - s.submitted;

    if (table_left <= 0)
    {
        printf("The task table is full (%d tasks). Use option 7 (Reset) to start over.\n",
               MAX_TASKS);
        return;
    }
    if (free_slots <= 0)
    {
        printf("The task queue is full (%d/%d).\n", s.queued, s.capacity);
        printf("Run option 4 (Start Task Processing) to drain the queue first.\n");
        return;
    }

    int limit = free_slots < table_left ? free_slots : table_left;

    printf("Free queue slots: %d of %d\n", free_slots, s.capacity);

    int how_many = (int)prompt_ll("How many tasks do you want to submit", 1, limit);

    for (int i = 1; i <= how_many; i++)
    {
        Task task;
        int  id = 0;

        printf("\n--- Task %d of %d ---", i, how_many);
        build_task_interactive(&task);

        SubmitResult r = pool_submit(&g_pool, &task, 0, &id);

        if (r == SUBMIT_OK)
        {
            char params[PARAM_LEN];
            describe_params(&task, params, sizeof(params));
            printf("Task #%d (%s, %s) added to the queue.\n",
                   id, task_type_name(task.type), params);
        }
        else
        {
            printf("Submission failed: %s\n",
                   r == SUBMIT_QUEUE_FULL ? "queue is full" :
                   r == SUBMIT_TABLE_FULL ? "task table is full" :
                                            "pool is shutting down");
            break;
        }
    }

    s = pool_snapshot(&g_pool);
    printf("\n");
    print_pool_summary(&s);
}

/* ================================================================
   FEATURE 3: DISPLAY TASK QUEUE
   ================================================================ */

static void display_task_queue(void)
{
    print_header("3. TASK QUEUE");

    if (!g_pool.configured)
    {
        printf("The thread pool is not configured. Use option 1 first.\n");
        return;
    }

    ThreadPool *p = &g_pool;

    pthread_mutex_lock(&p->lock);

    printf("Queue usage: %d / %d slots\n\n", p->count, p->queue_capacity);

    if (p->count == 0)
    {
        printf("The queue is empty.\n");
    }
    else
    {
        printf("%-9s %-5s %-11s %-34s %s\n",
               "Position", "ID", "Type", "Parameters", "Status");
        print_line();

        for (int k = 0; k < p->count; k++)
        {
            int         index = p->queue[(p->head + k) % p->queue_capacity];
            const Task *t = &p->tasks[index];
            char        params[PARAM_LEN];

            describe_params(t, params, sizeof(params));
            printf("%-9d %-5d %-11s %-34s %s%s\n",
                   k + 1, t->id, task_type_name(t->type), params,
                   status_name(t->status), k == 0 ? "   <-- next to run" : "");
        }
    }

    pthread_mutex_unlock(&p->lock);
}

/* ================================================================
   FEATURE 4: START TASK PROCESSING
   ================================================================ */

static void start_task_processing(void)
{
    print_header("4. START TASK PROCESSING");

    if (!g_pool.configured)
    {
        printf("The thread pool is not configured. Use option 1 first.\n");
        return;
    }

    PoolSnapshot s = pool_snapshot(&g_pool);

    if (s.pending == 0)
    {
        printf("There are no pending tasks. Submit tasks with option 2 first.\n");
        return;
    }

    int trace = (s.pending <= 50);

    printf("Releasing %d worker thread(s) to process %d pending task(s)...\n\n",
           s.workers, s.pending);
    if (!trace)
        printf("(More than 50 tasks: per-task messages are suppressed.)\n\n");

    double start = now_ms();

    pool_begin_processing(&g_pool, trace);   /* wake the workers        */
    pool_wait_idle(&g_pool);                 /* wait for all to finish  */
    pool_pause_processing(&g_pool);          /* workers sleep again     */

    double wall = now_ms() - start;

    printf("\nAll pending tasks have been processed in %.2f ms.\n", wall);
    printf("The worker threads are still alive and can be reused.\n");

    s = pool_snapshot(&g_pool);
    print_pool_summary(&s);
    print_worker_stats(&g_pool);
}

/* ================================================================
   FEATURE 5: DISPLAY TASK STATUS AND RESULTS
   ================================================================ */

static void display_status_and_results(void)
{
    print_header("5. TASK STATUS AND RESULTS");

    if (!g_pool.configured)
    {
        printf("The thread pool is not configured. Use option 1 first.\n");
        return;
    }

    PoolSnapshot s = pool_snapshot(&g_pool);

    print_pool_summary(&s);
    printf("\n");
    print_task_table(&g_pool);

    if (s.submitted > 0)
        print_worker_stats(&g_pool);
}

/* ================================================================
   FEATURE 6: RUN COMPLETE SIMULATION
   ================================================================ */

/* Small deterministic PRNG (xorshift64*) so a seed reproduces a run */
static unsigned long long rng_next(unsigned long long *state)
{
    unsigned long long x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static long long rng_range(unsigned long long *state, long long lo, long long hi)
{
    unsigned long long span = (unsigned long long)(hi - lo) + 1ULL;

    return lo + (long long)(rng_next(state) % span);
}

/* Build one random task whose size depends on the chosen intensity */
static void generate_task(Task *t, TaskType type, int intensity,
                          unsigned long long *rng)
{
    static const long long sum_max[3]   = {100000LL, 10000000LL, 100000000LL};
    static const long long prime_max[3] = {1000000LL, 1000000000LL, 1000000000000LL};
    static const long long work_max[3]  = {100000LL, 5000000LL, 50000000LL};
    static const long long list_val[3]  = {1000LL, 1000000LL, 1000000000LL};

    int lv = intensity - 1;

    memset(t, 0, sizeof(*t));
    t->type = type;

    switch (type)
    {
    case TASK_SUM:
        t->param = rng_range(rng, 1, sum_max[lv]);
        break;
    case TASK_FACTORIAL:
        t->param = rng_range(rng, 0, FACT_MAX_N);
        break;
    case TASK_PRIME:
        /* odd numbers force real trial division work */
        t->param = rng_range(rng, 1, (prime_max[lv] - 1) / 2) * 2 + 1;
        break;
    case TASK_LIST_STATS:
        t->list_count = (int)rng_range(rng, 5, 50);
        for (int i = 0; i < t->list_count; i++)
            t->list[i] = rng_range(rng, -list_val[lv], list_val[lv]);
        break;
    case TASK_WORKLOAD:
        t->param = rng_range(rng, 1, work_max[lv]);
        break;
    }
}

static void run_complete_simulation(void)
{
    print_header("6. RUN COMPLETE SIMULATION");

    /* A simulation builds its own pool, so an existing one is replaced */
    if (g_pool.configured)
    {
        PoolSnapshot cur = pool_snapshot(&g_pool);

        if (cur.processing || cur.running > 0)
        {
            printf("Cannot start: worker threads are currently active.\n");
            return;
        }
        if (!prompt_yes_no("The simulation REPLACES the current pool and its tasks. Continue?"))
        {
            printf("Simulation cancelled.\n");
            return;
        }
        pool_teardown(&g_pool);
        printf("Previous thread pool shut down.\n\n");
    }

    int workers   = (int)prompt_ll("Number of worker threads", 1, MAX_WORKERS);
    int capacity  = (int)prompt_ll("Maximum task queue capacity", 1, MAX_QUEUE_CAPACITY);
    int num_tasks = (int)prompt_ll("Number of tasks to submit", 1, MAX_TASKS);

    printf("\nTask mix:\n");
    printf("  1. Sum 1..N only        4. List statistics only\n");
    printf("  2. Factorial only       5. Simulated workload only\n");
    printf("  3. Prime check only     6. Random mix of all types\n");
    int mix = (int)prompt_ll("Select task mix", 1, 6);

    printf("\nWorkload intensity:  1 = light   2 = medium   3 = heavy\n");
    int intensity = (int)prompt_ll("Select intensity", 1, 3);

    if (intensity == 3 && num_tasks > 100)
        printf("Note: heavy workloads with many tasks may take a while.\n");

    long long seed_in = prompt_ll("Random seed (0 = use current time)", 0, 999999999LL);

    unsigned long long seed = (unsigned long long)seed_in;
    if (seed == 0)
        seed = (unsigned long long)time(NULL);

    unsigned long long rng = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    if (rng == 0)
        rng = 88172645463325252ULL;

    printf("\nCreating thread pool (%d workers, queue capacity %d)...\n",
           workers, capacity);

    if (pool_init(&g_pool, workers, capacity) != 0)
    {
        printf("Thread pool creation FAILED. Simulation aborted.\n");
        return;
    }

    int trace = (num_tasks <= 40);

    printf("Seed used: %llu\n", seed);
    printf("The main thread acts as PRODUCER; the %d workers are CONSUMERS.\n",
           workers);
    if (num_tasks > capacity)
        printf("Queue capacity (%d) is smaller than the task count (%d): the producer will\n"
               "block on a condition variable whenever the queue is full.\n",
               capacity, num_tasks);
    if (!trace)
        printf("(More than 40 tasks: per-task messages are suppressed.)\n");
    printf("\n");

    double start = now_ms();

    /* Workers start consuming immediately while we are still producing */
    pool_begin_processing(&g_pool, trace);

    int submitted_ok = 0;

    for (int i = 0; i < num_tasks; i++)
    {
        Task         task;
        int          id = 0;
        TaskType     type = (mix == 6) ? (TaskType)rng_range(&rng, 1, 5)
                                       : (TaskType)mix;

        generate_task(&task, type, intensity, &rng);

        /* blocking = 1: waits on 'not_full' if the queue is full */
        SubmitResult r = pool_submit(&g_pool, &task, 1, &id);

        if (r != SUBMIT_OK)
        {
            log_event("  [Producer] submission of task %d failed\n", i + 1);
            break;
        }
        submitted_ok++;

        if (trace)
        {
            char params[PARAM_LEN];
            describe_params(&task, params, sizeof(params));
            log_event("  [Producer] queued task #%-4d %-10s %s\n",
                      id, task_type_name(type), params);
        }
    }

    pool_wait_idle(&g_pool);             /* wait until all tasks finish */
    pool_pause_processing(&g_pool);

    double wall = now_ms() - start;

    printf("\nSimulation finished: %d task(s) submitted, wall-clock time %.2f ms.\n",
           submitted_ok, wall);

    PoolSnapshot s = pool_snapshot(&g_pool);
    print_pool_summary(&s);
    print_worker_stats(&g_pool);

    /* Sum of per-task times vs wall time gives an estimate of overlap */
    double total_task_ms = 0.0;

    pthread_mutex_lock(&g_pool.lock);
    for (int i = 0; i < g_pool.total_tasks; i++)
        total_task_ms += g_pool.tasks[i].exec_ms;
    pthread_mutex_unlock(&g_pool.lock);

    printf("\nTotal task execution time : %.2f ms\n", total_task_ms);
    printf("Wall-clock time           : %.2f ms\n", wall);
    if (wall > 0.0)
        printf("Estimated concurrency     : %.2fx (task time / wall time)\n",
               total_task_ms / wall);

    if (num_tasks <= 30)
    {
        printf("\nResults:\n");
        print_task_table(&g_pool);
    }
    else
    {
        printf("\nUse option 5 to view the status and result of every task.\n");
    }
}

/* ================================================================
   FEATURE 7: RESET THREAD POOL
   ================================================================ */

static void reset_thread_pool(void)
{
    print_header("7. RESET THREAD POOL");

    if (!g_pool.configured)
    {
        printf("Nothing to reset: the thread pool is not configured.\n");
        return;
    }

    PoolSnapshot s = pool_snapshot(&g_pool);

    if (s.processing || s.running > 0)
    {
        printf("Cannot reset: worker threads are currently active.\n");
        return;
    }

    print_pool_summary(&s);

    if (!prompt_yes_no("Stop all workers and discard the queue and all task records?"))
    {
        printf("Reset cancelled.\n");
        return;
    }

    pool_teardown(&g_pool);
    printf("All worker threads joined, memory released.\n");
    printf("The thread pool is now unconfigured. Use option 1 to create a new one.\n");
}

/* ================================================================
   MAIN MENU
   ================================================================ */

static void show_main_menu(void)
{
    print_header("THREAD POOL BASED TASK PROCESSING SYSTEM");

    if (g_pool.configured)
    {
        PoolSnapshot s = pool_snapshot(&g_pool);
        printf(" Pool: %d worker(s), queue %d/%d | submitted %d, done %d, pending %d, failed %d\n",
               s.workers, s.queued, s.capacity,
               s.submitted, s.completed, s.pending, s.failed);
    }
    else
    {
        printf(" Pool: NOT CONFIGURED\n");
    }
    print_line();

    printf(" 1. Configure Thread Pool\n");
    printf(" 2. Submit Tasks\n");
    printf(" 3. Display Task Queue\n");
    printf(" 4. Start Task Processing\n");
    printf(" 5. Display Task Status and Results\n");
    printf(" 6. Run Complete Simulation\n");
    printf(" 7. Reset Thread Pool\n");
    printf(" 8. Exit\n");
    print_line();
}

int main(void)
{
    for (;;)
    {
        show_main_menu();

        int choice = (int)prompt_ll("Enter your choice", 1, 8);

        switch (choice)
        {
        case 1: configure_thread_pool();       break;
        case 2: submit_tasks();                break;
        case 3: display_task_queue();          break;
        case 4: start_task_processing();       break;
        case 5: display_status_and_results();  break;
        case 6: run_complete_simulation();     break;
        case 7: reset_thread_pool();           break;

        case 8:
        {
            if (g_pool.configured)
            {
                PoolSnapshot s = pool_snapshot(&g_pool);

                if (s.pending > 0 &&
                    !prompt_yes_no("There are unprocessed tasks that will be discarded. Exit anyway?"))
                    continue;
            }

            printf("\nStopping worker threads and releasing resources...\n");
            pool_teardown(&g_pool);
            printf("All threads joined. Goodbye!\n");
            return 0;
        }
        }

        pause_screen();
    }
}
