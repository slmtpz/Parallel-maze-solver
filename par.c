/*
Student Name: Abdullah Seleme Topuz
Student Number: 2012400111
Compile Status: Compiling
Program Status: Working
Notes: Want to mention that in Ubuntu, txt files I obtain will be perfectly same with output files of yours whereas when I email them and open via Windows, new lines are ignored.
*/
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
/*
#define INPUTFILEPATH "io/in1.txt"
#define OUTPUTFILEPATH "o1.txt"
*/
int **allocMaze (int rows, int cols);

void iterateSeq (int **maze, int n, int p);

void print_grid (int n, int p, int myrank, int **grid);

void communicate (int rank, int *sendToBot, int *sendToTop, int *recvFromBot, int *recvFromTop, int n, int p);

void prepareRequests (int *sendToBot, int *sendToTop, int n, int p, int **maze);

void prepareReplies (int *sendToBot, int *sendToTop, int *recvFromBot, int *recvFromTop, int n, int p, int **maze);

void useReplies (int *recvFromBot, int *recvFromTop, int n, int p, int **maze, int *done);

void deallocMaze (int **maze);

void main (int argc, char *argv[]) {

    /******************* INITIALIZATIONS AND DECLARATIONS **********************/
    /*
     * MPI Environment will be set.
     * The master processor with rank=0, will read input file and
     *   broadcast n value over all processors.
     * 1d and 2d arrays will be initialized.
     */
    int rank, n, p, i, j, done = 1, alldone = 1;
    FILE *fp;
    int **maze;
    MPI_Status status;

    MPI_Init(&argc, &argv); // initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank is set
    MPI_Comm_size(MPI_COMM_WORLD, &p);  // number of processors
    p-=1; // excluding master proc. from p value



    if (rank == 0) {
        fp=fopen(argv[1], "r");
        fscanf(fp, "%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);   // n value is read from input by master and broadcast to all slaves
    MPI_Barrier(MPI_COMM_WORLD);        // to make sure all got the broadcast before using n

    maze = allocMaze(n/p, n);  //to make the 2d-array contigous on memory

    int sendToBot[n], sendToTop[n], recvFromTop[n], recvFromBot[n];
    for (i=0; i<n; i++) {
        sendToBot[i]=0;
        sendToTop[i]=0;
        recvFromTop[i]=0;
        recvFromBot[i]=0;
    }


    /************** DISTRIBUTION OF INPUT OVER PROCESSORS *****************/
    /*
     * Master will read n*n/p elements for each processors and pack them
     *   into an array and send it to processors via MPI communication.
     */
    if (rank == 0) {
        int sr;
        for (sr=1; sr<=p; sr++) {
            for (i=0; i<n/p; i++) {
                for (j=0; j<n; j++) {
                    fscanf(fp, "%d", &(maze[i][j]));
                }
            }
            MPI_Send(&(maze[0][0]), n*n/p, MPI_INT, sr, 0, MPI_COMM_WORLD);
        }
        fclose(fp);
    } else {
        MPI_Recv(&(maze[0][0]), n*n/p, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /************************* MAIN LOOP ****************************/
    /*
     * All the processors including master go into a while loop.
     * The arrays sendToBot, sendToTop, recvFromBot, recvFromTop will
     *   act as buffers in the communication phases.
     * Slave processors' work in the while loop:
     * --> iterateSeq
     *     Run sequential maze solving algorithm in the n/p * n 2d-array
     *     called "maze" for each processor.
     * --> prepareRequests
     *     Fill the request arrays which are sendToTop and sendToBot to
     *     send to neighbor processors.
     * --> communicate
     *     All request arrays will be send to neighbor processors for all
     *     processors. A processor will take the requests into recvFromBot
     *     and recvFromTop arrays.
     * --> prepareReplies
     *     Processors will read the requests and prepare sendToTop and
     *     sendToBot to reply back.
     * --> communicate
     *     Processors will send their replies back to those that requested
     *     from them.
     * --> useReplies
     *     The arrays taken back as replies will be iterated and change
     *     the corresponding values in the mazes.
     * --> Communicate with the master and break of the while
     *     In each iteration of this while-loop slaves will send "done"
     *     values to the master indicating if they have changed any value
     *     in this iteration or not. If all of them had not changed any
     *     value in one iteration, it is sufficient to stop there.
     *
     * Master processor's work in the while loop:
     * --> Communicate with the slaves and break of the while
     *     In each iteration, master will receive their done values, and
     *     check if they are all done or not. If any of them had changed
     *     anything in an iteration, it is sufficient to let them continue.
     */
    if (rank != 0) {
        while(1) {
            //slave loop
            iterateSeq(maze, n, p);

            prepareRequests(sendToBot, sendToTop, n, p, maze);

            communicate(rank, sendToBot, sendToTop, recvFromBot, recvFromTop, n, p);

            prepareReplies(sendToBot, sendToTop, recvFromBot, recvFromTop, n, p, maze);

            communicate(rank, sendToBot, sendToTop, recvFromBot, recvFromTop, n, p);

            useReplies(recvFromBot, recvFromTop, n, p, maze, &done);

            MPI_Send(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&alldone, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (alldone == 1) break;
            done = 1;
        }
    } else {
        // master loop
        while(1) {
            int sr;
            alldone = 1;
            for (sr=1; sr<=p; sr++) {
                MPI_Recv(&done, 1, MPI_INT, sr, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                alldone = alldone && done;
            }
            for (sr=1; sr<=p; sr++) {
                MPI_Send(&alldone, 1, MPI_INT, sr, 0, MPI_COMM_WORLD);
            }
            if (alldone == 1) break;
        }
    }

    /******************** ENDING AND OUTPUT *************************/
    /*
     * Now that while loops are done, the solving of the maze is done.
     * All slaves will send their own mazes to the master.
     * Master will output the mazes.
     */
    if (rank != 0) {
        MPI_Send(&(maze[0][0]), n*n/p, MPI_INT, 0, 0, MPI_COMM_WORLD);
        deallocMaze(maze);
    } else {
        fp=fopen(argv[2], "w");
        int sr;
        for (sr=1; sr<=p; sr++) {
            MPI_Recv(&(maze[0][0]), n*n/p, MPI_INT, sr, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i=0; i<n/p; i++) {
                for (j=0; j<n; j++) {
                  //  if(maze[i][j]==1)
                    fprintf(fp, " %d", maze[i][j]);
                   // else
                    //    fprintf(fp, "  ");
                }
                fprintf(fp, "\n");
            }
        }
        fclose(fp);
        deallocMaze(maze);
    }

    //MPI_Barrier(MPI_COMM_WORLD);
    // printf("proc %d exiting\n", rank);
    MPI_Finalize(); // clean MPI
}

/*
 *  Allocates a 2d-array in the memory specifically since
 *  we want the mazes contiguous to use them on communication.
 */
int **allocMaze (int rows, int cols) {
    int *data=(int *) malloc(rows*cols*sizeof(int));
    int **array=(int **) malloc(rows*sizeof(int *));
    int i;
    for (i=0; i<rows; i++)
        array[i]=&(data[cols*i]);

    return array;
}

/*
 * Prints the maze onto command-line interface.
 * Can be used to test.
 * Caution: Use barriers between the prints of each mazes.
 */
void print_grid (int n, int p, int myrank, int **grid) {
    int i, j;
    printf("P%d's grid:\n", myrank);
    for (i=0; i<n/p; i++) {
        for (j=0; j<n; j++) {
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }
}

/*
 * Checks adjacent indexes of 2d (n*n/p) maze array.
 * Returns how many 0's (black cell) are there for an index.
 * --> Only checks in the array. Does not do communicate.
 */
int wallNeighbors (int **grid, int i, int j, int n, int p) {
    int result=0;
    if (i != 0) {
        // if it's not 1 st row, look upside
        if (grid[i-1][j] == 0) result++;
    }
    if (j != n-1) {
        // if it's not n th col, look rightside
        if (grid[i][j+1] == 0) result++;
    }
    if (i != n/p-1) {
        // if it's not n/p th row, look downside
        if (grid[i+1][j] == 0) result++;
    }
    if (j != 0) {
        // if it's not 1 st col, look leftside
        if (grid[i][j-1] == 0) result++;
    }
    return result;
}

/*
 * Sequential solving of the algorithm.
 * Makes white cells containing 3 black cells around
 * a black cell.
 */
void iterateSeq (int **maze, int n, int p) {
    int i, j;
    char no_more_dead_ends=0;
    while (!no_more_dead_ends) {
        no_more_dead_ends=1;
        for (i=0; i<n/p; i++) {
            for (j=0; j<n; j++) {
                if (maze[i][j] == 1) {
                    if (wallNeighbors(maze, i, j, n, p) == 3) {
                        maze[i][j]=0;
                        no_more_dead_ends=0;
                    }
                }
            }
        }
    }
}

/*
 * Send and Recv calls of MPI used to communicate among processors.
 *
 * All processors are divided into 2 groups that are odd ones and even ones
 *   depending on their ranks.
 * First, odd ones will send their requests to even ones.
 * In the mean time, even ones will recv the requests from odd ones.
 * And then, even ones will send requests to odd ones.
 * In the mean time, odd ones will recv the requests from even ones.
 *
 * Deadlocks are avoided since all the processors sends and receives at the
 * same time. There will not be a case that sends will oversize recvs or vice versa.
 *
 * First processor will not send to top, and recv from top; last processor will not
 *   send to bot, and recv from bot.
 */
void communicate (int rank, int *sendToBot, int *sendToTop, int *recvFromBot, int *recvFromTop, int n, int p) {
    if (rank%2 == 1) {
        if (rank != p)
            MPI_Send(sendToBot, n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        if (rank != 1)
            MPI_Send(sendToTop, n, MPI_INT, rank-1, 0, MPI_COMM_WORLD);

        if (rank != 1)
            MPI_Recv(recvFromTop, n, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank != p)
            MPI_Recv(recvFromBot, n, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(recvFromTop, n, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank != p)
            MPI_Recv(recvFromBot, n, MPI_INT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank != p)
            MPI_Send(sendToBot, n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        MPI_Send(sendToTop, n, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
    }
}

/*
 * Initializes sendToBot and sendToTop arrays to send to adjacent processors.
 * If there are 1's on the border rows and if their number of black cell neighbors
 *   are 2, the cell may be a dead end, hence the corresponding index of the
 *   buffer array is set to 1, other ones 0.
 */
void prepareRequests (int *sendToBot, int *sendToTop, int n, int p, int **maze) {
    int j;
    for (j=0; j<n; j++) {
        if (maze[n/p-1][j] == 1 && wallNeighbors(maze, n/p-1, j, n, p) == 2)
            sendToBot[j]=1;
        else
            sendToBot[j]=0;
    }
    for (j=0; j<n; j++) {
        if (maze[0][j] == 1 && wallNeighbors(maze, 0, j, n, p) == 2)
            sendToTop[j]=1;
        else
            sendToTop[j]=0;
    }
}

/*
 * After communication processors received request buffers. Processors check the
 *   indexes of the request buffers and if there are any 1, it means the requester want
 *   to know the corresponding index of the border row. If it is 0, it means that the
 *   requesters index will be dead end since it has 2 black cell neighbors already.
 *   Otherwise, it is not a dead end.
 */
void prepareReplies (int *sendToBot, int *sendToTop, int *recvFromBot, int *recvFromTop, int n, int p, int **maze) {
    int j;
    for (j=0; j<n; j++) {
        if (recvFromTop[j] == 1 && maze[0][j] == 0)
            sendToTop[j]=1;  // 1 mean its wallN = 3
        else
            sendToTop[j]=0;  // 0 mean its wallN = 2

        if (recvFromBot[j] == 1 && maze[n/p-1][j] == 0)
            sendToBot[j]=1;
        else
            sendToBot[j]=0;
    }
}

/*
 * Checks the reply buffers if there are any 1. If there is any, the corresponding
 *   index will be set to 0, a dead-end, black cell.
 * If there is any change in the maze, change done variable to 0, so that the processor
 *   is not done yet, will ask for another iteration of the while loop.
 */
void useReplies (int *recvFromBot, int *recvFromTop, int n, int p, int **maze, int *done) {
    int j;
    for (j=0; j<n; j++) {
        if (recvFromTop[j] == 1) {
            maze[0][j]=0;
            *done = 0;
        }
        if (recvFromBot[j] == 1) {
            maze[n/p-1][j]=0;
            *done = 0;
        }
    }
}

/*
 * Deallocate the memory since maze will not be used anymore.
 */
void deallocMaze (int **maze) {
    free(maze[0]);
    free(maze);
}