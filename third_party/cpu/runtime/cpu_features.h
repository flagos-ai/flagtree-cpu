// Runtime CPU feature probes for the TLE ARM64 kernels.
//
// Frozen C ABI: out-of-tree consumers (e.g. llama.cpp's flaggems backend)
// link these symbols directly. Bump TLE_RUNTIME_ABI_VERSION whenever any
// extern "C" signature in this runtime changes incompatibly.
#pragma once

#define TLE_RUNTIME_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

// Each returns 1 if the feature is usable (hardware supports it AND it is
// not disabled), else 0. Results are probed once and cached.
//
// Environment overrides can only DISABLE a feature, never force one on:
//   TLE_CPU_DISABLE_DOTPROD=1  TLE_CPU_DISABLE_I8MM=1  TLE_CPU_DISABLE_SME=1
// (disabling SME also disables SME2).
int tle_cpu_has_dotprod(void);
int tle_cpu_has_i8mm(void);
int tle_cpu_has_sme(void);
int tle_cpu_has_sme2(void);

int tle_runtime_abi_version(void);

#ifdef __cplusplus
}
#endif
