// Helper functions to call for the mkl
//
// Author: Saman Biookaghazadeh

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mkl.h"

#define ABS(a)		(((a) < 0) ? -(a) : (a))
#ifdef MKL_ILP64
	#define INT_FORMAT "%lld"
#else
	#define INT_FORMAT "%d"
#endif


typedef struct mkl_config {
	MKL_INT		m, n, lda, incx, incy;
	MKL_INT		rmaxa, cmaxa;
	float 		alpha, beta;
	float		*a, *x, *y;
	CBLAS_LAYOUT	layout;
	CBLAS_TRANSPOSE	trans;
	MKL_INT		nx, ny, len_x, len_y;
};

int init_mkl_config(struct mkl_config* config, int m, int n);
void warmup_mkl(struct mkl_config* config, int iterations);
void run_mkl(struct mkl_config* config);
void finish_mkl(struct mkl_config* config);

int init_mkl_config(struct mkl_config* config, int m, int n) {

	config->n = n;
	config->m = m;
	config->layout = CblasRowMajor;
	config->trans = CblasNoTrans;
	config->alpha = 1.0;
	config->beta = 1.0;
	config->incx = 1;
	config->incy = 1;
	config->rmaxa = m + 1;
	config->cmaxa = n;
	
	config->a = (float *) mkl_calloc(config->rmaxa * config->cmaxa, sizeof(float), 64);
	
	config->nx = n;
	config->ny = m;
	
	config->len_x = 1+(n-1)*ABS(config->incx);
	config->len_y = 1+(m-1)*ABS(config->incy);

	config->x = (float *) mkl_calloc(config->len_x, sizeof(float), 64);
	config->y = (float *) mkl_calloc(config->len_y, sizeof(float), 64);

	if (config->a == NULL || config->x == NULL || config->y == NULL) {
		printf("\n Can't allocate memory for arrays\n");
		mkl_free(config->a);
		mkl_free(config->x);
		mkl_free(config->y);

		return 0;
	}

	config->lda = config->cmaxa;
	
	printf("\n	INPUT_DATA");
	printf("\n	M="INT_FORMAT"	N="INT_FORMAT, config->m, config->n);
	printf("\n	ALPHA=%5.1f	BETA=%5.1f", config->alpha, config->beta);	
	
	return 1;
}

void warmup_mkl(struct mkl_config* config, int iterations) {

	for (int i = 0; i < iterations; i++) {
		cblas_sgemv(config->layout, config->trans, config->m, config->n, 
				config->alpha, config->a, config->lda, config->x,
				config->incx, config->beta, config->y, config->incy);
	}

}

void run_mkl(struct mkl_config* config) {

	cblas_sgemv(config->layout, config->trans, config->m, config->n,
			config->alpha, config->a, config->lda, config->x,
			config->incx, config->beta, config->y, config->incy);
}

void finish_mkl(struct mkl_config* config) {
	
	mkl_free(config->a);
	mkl_free(config->x);
	mkl_free(config->y);

}
