#pragma once

#define BIN_NPHI 32
#define BIN_NTHETA 9
#define MAX_THETA_BINS 100

#define SH_NTHETA 64
#define SH_NPHI 128
#define SH_STRIDE 8192

#define MAX_SH_ORDER 150

__constant__ float g_theta_bins[MAX_THETA_BINS];