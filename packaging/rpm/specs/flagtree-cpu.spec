# Wraps a pre-built flagtree_cpu wheel (see packaging/rpm/helpers) into an rpm.
%{!?wheel_dir: %global wheel_dir %{_sourcedir}}
%define _python_bytecompile_errors_terminate_build 0
%global debug_package %{nil}

Name:           python3-flagtree-cpu
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagTree compiler with CPU backend (Triton-CPU)

License:        MIT AND Apache-2.0 WITH LLVM-exception AND BSL-1.0
URL:            https://github.com/flagos-ai/flagtree-cpu
ExclusiveArch:  aarch64 x86_64
Source0:        LICENSE

BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-setuptools
BuildRequires:  python3-wheel

Requires:       python3
# Runtime, not only build time: the CPU backend compiles a C++/OpenMP kernel
# launcher against Python.h at the first kernel launch.
Requires:       gcc-c++
Requires:       python3-devel
Recommends:     python3-numpy

Provides:       python3-flagtree
Provides:       python3-triton
Conflicts:      python3-triton
Conflicts:      python3-flagtree-nvidia
Conflicts:      python3-flagtree-iluvatar
Conflicts:      python3-flagtree-mthreads
Conflicts:      python3-flagtree-metax
Conflicts:      python3-flagtree-amd
Conflicts:      python3-flagtree-cambricon
Conflicts:      python3-flagtree-hcu
Conflicts:      python3-flagtree-xpu

%description
flagtree-cpu is the FlagOS CPU backend for the Triton language, derived from
triton-lang/triton-cpu. It compiles Triton kernels to native code for the
host CPU (AArch64 with NEON/SVE2/I8MM, x86-64 with AVX) and runs them
through OpenMP.

This package installs into the "triton" Python namespace, like the other
FlagTree backend packages, and therefore conflicts with them and with the
upstream python3-triton package. Select the backend at runtime with
TRITON_CPU_BACKEND=1.

%prep

%build

%install
set -e
WHEEL=$(ls %{wheel_dir}/flagtree_cpu-*.whl 2>/dev/null | head -n1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: no flagtree_cpu-*.whl found under %{wheel_dir}" >&2
    exit 1
fi
PYDIR=%{buildroot}%{python3_sitearch}
mkdir -p "$PYDIR" %{buildroot}%{_bindir}
python3 -m pip install --no-deps --no-compile --no-index \
    --target="$PYDIR" \
    "$WHEEL"
if [ -d "$PYDIR/bin" ]; then
    mv "$PYDIR/bin/"* %{buildroot}%{_bindir}/ 2>/dev/null || true
    rmdir "$PYDIR/bin" 2>/dev/null || true
fi
rm -f "$PYDIR"/flagtree_cpu-*.dist-info/RECORD
# The compile-cache key hashes triton/_C/libtriton.so; make sure that exact
# name exists even when the wheel only carries the suffixed one.
if [ ! -e "$PYDIR"/triton/_C/libtriton.so ]; then
    suffixed="$(ls -1 "$PYDIR"/triton/_C/libtriton.cpython-*.so 2>/dev/null | head -1)"
    test -n "$suffixed"
    mv "$suffixed" "$PYDIR"/triton/_C/libtriton.so
fi
test -f "$PYDIR"/triton/_C/libtriton.so
install -D -m 0644 %{SOURCE0} %{buildroot}%{_licensedir}/%{name}/LICENSE

%files
%license %{_licensedir}/%{name}/LICENSE
%{python3_sitearch}/triton
%{python3_sitearch}/flagtree_cpu-*.dist-info

%changelog
* Sat Oct 17 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.
