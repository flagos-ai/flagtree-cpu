// Runtime CPU feature detection for the TLE ARM64 kernels.
// macOS: sysctlbyname("hw.optional.arm.FEAT_*"); Linux: getauxval(AT_HWCAP*).
// Probed once (thread-safe static init), env vars demote only.
#include "cpu_features.h"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

namespace {

bool env_disabled(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

#if defined(__APPLE__) && defined(__aarch64__)
bool sysctl_flag(const char *oid) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(oid, &value, &size, nullptr, 0) != 0)
    return false;
  return value != 0;
}
#endif

struct Features {
  bool dotprod = false;
  bool i8mm = false;
  bool sme = false;
  bool sme2 = false;

  Features() {
#if defined(__APPLE__) && defined(__aarch64__)
    dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
    sme = sysctl_flag("hw.optional.arm.FEAT_SME");
    sme2 = sysctl_flag("hw.optional.arm.FEAT_SME2");
#elif defined(__linux__) && defined(__aarch64__)
    // Numeric fallbacks for the HWCAP bits so older glibc headers still
    // compile: ASIMDDP=HWCAP bit 20; I8MM=HWCAP2 bit 13, SME=HWCAP2 bit 23,
    // SME2 via HWCAP2 bit 37 (linux >= 6.3 headers define HWCAP2_SME2).
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_ASIMDDP
    dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#else
    dotprod = (hwcap & (1UL << 20)) != 0;
#endif
#ifdef HWCAP2_I8MM
    i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#else
    i8mm = (hwcap2 & (1UL << 13)) != 0;
#endif
#ifdef HWCAP2_SME
    sme = (hwcap2 & HWCAP2_SME) != 0;
#else
    sme = (hwcap2 & (1UL << 23)) != 0;
#endif
#ifdef HWCAP2_SME2
    sme2 = (hwcap2 & HWCAP2_SME2) != 0;
#else
    sme2 = (hwcap2 & (1UL << 37)) != 0;
#endif
#endif // platform

#ifdef TLE_NO_SME_KERNEL
    // The SME tile kernel was not assembled into this build (toolchain lacks
    // SME2); report SME unusable regardless of hardware.
    sme = false;
    sme2 = false;
#endif

    if (env_disabled("TLE_CPU_DISABLE_DOTPROD")) {
      // dotprod is the baseline of the int8 stack; everything above it
      // depends on packed-int8 helpers compiled in the dotprod region.
      dotprod = false;
      i8mm = false;
      sme = false;
      sme2 = false;
    }
    if (env_disabled("TLE_CPU_DISABLE_I8MM"))
      i8mm = false;
    if (env_disabled("TLE_CPU_DISABLE_SME")) {
      sme = false;
      sme2 = false;
    }
  }
};

const Features &features() {
  static Features f;
  return f;
}

} // namespace

extern "C" {

EXPORT int tle_cpu_has_dotprod(void) { return features().dotprod ? 1 : 0; }
EXPORT int tle_cpu_has_i8mm(void) { return features().i8mm ? 1 : 0; }
EXPORT int tle_cpu_has_sme(void) { return features().sme ? 1 : 0; }
EXPORT int tle_cpu_has_sme2(void) { return features().sme2 ? 1 : 0; }

EXPORT int tle_runtime_abi_version(void) { return TLE_RUNTIME_ABI_VERSION; }

} // extern "C"
