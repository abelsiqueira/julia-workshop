#include <stdio.h>
#include <stdlib.h>

void update_bfgs(int n, double *B, double *y, double *s) {
  int i, j, k;
  double *Bs, a = 0.0, b = 0.0;

  for (i = 0; i < n; i++)
    a += y[i]*s[i];

  if (a <= 0)
    return;

  Bs = (double *) malloc(sizeof(double)*n);

  for (i = 0; i < n; i++) {
    Bs[i] = 0.0;
    for (j = 0; j < n; j++)
      Bs[i] += B[i + j*n]*s[j];
    b += s[i]*Bs[i];
  }

  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      B[i + j*n] += y[i]*y[j]/a - Bs[i]*Bs[j]/b;
      B[j + i*n] = B[i + j*n];
    }
  }

  free(Bs);

}

void update_bfgs2(int n, double *B, double *y, double *s, double *Bs) {
  int i, j, k;
  double a = 0.0, b = 0.0;

  for (i = 0; i < n; i++)
    a += y[i]*s[i];

  if (a <= 0)
    return;

  for (i = 0; i < n; i++) {
    Bs[i] = 0.0;
    for (j = 0; j < n; j++)
      Bs[i] += B[i + j*n]*s[j];
    b += s[i]*Bs[i];
  }

  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      B[i + j*n] += y[i]*y[j]/a - Bs[i]*Bs[j]/b;
      B[j + i*n] = B[i + j*n];
    }
  }

}
