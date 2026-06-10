//requires additional changes to the code to make it work

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static double gtod_ref_time_sec = 0.0;


double ge_flop(int SIZE)
{
    double n = (double) SIZE;
    return (n - 1.0) * n * (2.0 * n - 1.0) / 2.0;
}
/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock()
{
  double the_time, norm_sec;
  struct timeval tv;
  gettimeofday( &tv, NULL );
  if ( gtod_ref_time_sec == 0.0 )
    gtod_ref_time_sec = ( double ) tv.tv_sec;
  norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
  the_time = norm_sec + tv.tv_usec * 1.0e-6;
  return the_time;
}

int ge(double ** A, int SIZE)
{
  int i,j,k;
  double multipler;
  for (k = 0; k < SIZE; k++) { 
    for (i = k+1; i < SIZE; i++) { 
      multipler = (A[i][k] / A[k][k]);
      for (j = k+1; j < SIZE; j++) { 
         A[i][j] = A[i][j]-A[k][j]*multipler;
      } 
    }
  }
  return 0;
}

int main( int argc, const char* argv[] )
{
  int i,j,k,iret;
  double dtime;
  int SIZE = atoi(argv[1]);
  double ** matrix;
  double * matrix_;
  matrix_ = (double *) malloc(SIZE*SIZE*sizeof(double));
  matrix = (double **) malloc(SIZE*sizeof(double));

  // double matrix[SIZE][SIZE]; // TODO - make near optimal dynamic allocation
  for (i=0; i < SIZE; i++){
    matrix[i] = matrix_ + i*SIZE;
  }
  srand(1);
  for (i = 0; i < SIZE; i++) { 
    for (j = 0; j < SIZE; j++) { 
      matrix[i][j] = rand();
    }
  }
  printf("call GE");
  dtime = dclock();
  iret = ge(matrix, SIZE);
  dtime = dclock()-dtime;
  printf("Time: %le \n", dtime);
  printf("FLOP: %le \n", ge_flop(SIZE));
  printf("GFLOP/s: %le \n", ge_flop(SIZE) / dtime / 1.0e9);

  double check=0.0;
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      check = check + matrix[i][j];
    }
  }
  printf( "Check: %le \n", check);
  fflush( stdout );


  return iret;
}

