// Sub-grid terrain sampling support for ANUGA
//
// Provides inline functions for:
//   - Piecewise-linear table interpolation (volume from stage, area from stage)
//   - Volume inversion (stage from volume) via binary search
//
// Tables are stored as flat arrays with per-cell offsets:
//   sg_cell_eta[offset..offset+n_levels-1]  = elevation breakpoints
//   sg_cell_volume[offset..offset+n_levels-1] = cumulative volume
//   sg_cell_wet_area[offset..offset+n_levels-1] = wet planform area
//
// This header is included by sw_domain_openmp.c

#ifndef SW_SUBGRID_H
#define SW_SUBGRID_H

#include <math.h>

// ---- Piecewise-linear interpolation on a sorted table ----

// Binary search: find index i such that table[i] <= val < table[i+1]
// Returns 0 if val <= table[0], n-2 if val >= table[n-1]
static inline int sg_bisect(const double * __restrict table, int n, double val) {
    if (val <= table[0]) return 0;
    if (val >= table[n - 1]) return n - 2;

    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (table[mid] <= val) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Interpolate value from a piecewise-linear table
// x_table[n], y_table[n] define the breakpoints
// Returns y(x) via linear interpolation, with linear extrapolation
// beyond the table range using the last segment slope.
static inline double sg_interp(
    const double * __restrict x_table,
    const double * __restrict y_table,
    int n,
    double x)
{
    if (n <= 0) return 0.0;
    if (n == 1) return y_table[0];

    if (x <= x_table[0]) {
        return y_table[0];
    }

    if (x >= x_table[n - 1]) {
        // Linear extrapolation using last segment slope
        double dx = x_table[n - 1] - x_table[n - 2];
        if (dx > 1.0e-30) {
            double slope = (y_table[n - 1] - y_table[n - 2]) / dx;
            return y_table[n - 1] + slope * (x - x_table[n - 1]);
        }
        return y_table[n - 1];
    }

    int i = sg_bisect(x_table, n, x);
    double dx = x_table[i + 1] - x_table[i];
    if (dx < 1.0e-30) return y_table[i];
    double t = (x - x_table[i]) / dx;
    return y_table[i] + t * (y_table[i + 1] - y_table[i]);
}

// Invert a monotonically non-decreasing piecewise-linear function:
// Given y, find x such that f(x) = y, where f is defined by
// x_table, y_table breakpoints.
// For extrapolation above the table, uses the last segment slope
// (which equals cell_area for volume tables).
static inline double sg_invert(
    const double * __restrict x_table,
    const double * __restrict y_table,
    int n,
    double y,
    double extrap_slope)
{
    if (n <= 0) return 0.0;
    if (n == 1) return x_table[0];

    if (y <= y_table[0]) {
        return x_table[0];
    }

    if (y >= y_table[n - 1]) {
        // Linear extrapolation above table
        if (extrap_slope > 1.0e-30) {
            return x_table[n - 1] + (y - y_table[n - 1]) / extrap_slope;
        }
        return x_table[n - 1];
    }

    // Binary search on y_table
    int i = sg_bisect(y_table, n, y);
    double dy = y_table[i + 1] - y_table[i];
    if (dy < 1.0e-30) return x_table[i];
    double t = (y - y_table[i]) / dy;
    return x_table[i] + t * (x_table[i + 1] - x_table[i]);
}


// ---- Convenience functions using the domain's sub-grid arrays ----

// Get volume V(eta) for cell k
// Uses cell_area for above-table extrapolation (consistent with Python)
static inline double sg_volume_from_stage(
    const double * __restrict sg_eta,
    const double * __restrict sg_vol,
    const int * __restrict sg_offset,
    const int * __restrict sg_n_levels,
    const double * __restrict sg_cell_area,
    int k,
    double eta)
{
    int off = sg_offset[k];
    int n = sg_n_levels[k];
    if (n <= 0) return 0.0;

    const double *x = sg_eta + off;
    const double *y = sg_vol + off;

    if (eta <= x[0]) return 0.0;

    if (eta >= x[n - 1]) {
        // Extrapolate above table using cell_area (matches Python)
        return y[n - 1] + sg_cell_area[k] * (eta - x[n - 1]);
    }

    // In-table interpolation
    int i = sg_bisect(x, n, eta);
    double dx = x[i + 1] - x[i];
    if (dx < 1.0e-30) return y[i];
    double t = (eta - x[i]) / dx;
    return y[i] + t * (y[i + 1] - y[i]);
}

// Get wet area A_w(eta) for cell k
// Returns cell_area above table range (consistent with Python)
static inline double sg_wet_area_from_stage(
    const double * __restrict sg_eta,
    const double * __restrict sg_area,
    const int * __restrict sg_offset,
    const int * __restrict sg_n_levels,
    const double * __restrict sg_cell_area,
    int k,
    double eta)
{
    int off = sg_offset[k];
    int n = sg_n_levels[k];
    if (n <= 0) return sg_cell_area[k]; // fallback to full area

    const double *x = sg_eta + off;
    const double *y = sg_area + off;

    if (eta <= x[0]) return 0.0;
    if (eta >= x[n - 1]) return sg_cell_area[k];

    // In-table interpolation
    int i = sg_bisect(x, n, eta);
    double dx = x[i + 1] - x[i];
    if (dx < 1.0e-30) return y[i];
    double t = (eta - x[i]) / dx;
    return y[i] + t * (y[i + 1] - y[i]);
}

// Invert: get stage eta from volume V for cell k
static inline double sg_stage_from_volume(
    const double * __restrict sg_eta,
    const double * __restrict sg_vol,
    const int * __restrict sg_offset,
    const int * __restrict sg_n_levels,
    const double * __restrict sg_cell_area,
    int k,
    double V)
{
    int off = sg_offset[k];
    int n = sg_n_levels[k];
    if (n <= 0) return 0.0;
    return sg_invert(sg_eta + off, sg_vol + off, n, V, sg_cell_area[k]);
}

#endif // SW_SUBGRID_H
