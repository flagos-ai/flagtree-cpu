# Installing python3-flagtree-cpu

```bash
# Debian / Ubuntu
sudo apt install ./python3-flagtree-cpu_*.deb
# Fedora / openEuler
sudo dnf install ./python3-flagtree-cpu-*.rpm
```

The package installs the Triton compiler into the `triton` Python namespace
with the CPU backend built in. It therefore conflicts with the upstream
`python3-triton` package and with the other `python3-flagtree-<backend>`
packages; only one of them can be installed at a time.

Select the CPU backend at runtime:

```bash
export TRITON_CPU_BACKEND=1
```

The backend compiles a small C++/OpenMP launcher on the first kernel launch,
which is why the package depends on `g++` and the Python development headers.

Torch is needed by most Triton programs but is intentionally **not** declared
as a dependency: install the CPU build that matches your Python via pip,
preferably in a venv that can still see the system packages:

```bash
python3 -m venv --system-site-packages ~/.venv/flagos
source ~/.venv/flagos/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Smoke test:

```bash
TRITON_CPU_BACKEND=1 python3 - <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 1 << 16
x, y = torch.rand(n), torch.rand(n)
o = torch.empty_like(x)
add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
assert torch.allclose(o, x + y)
print("ok")
PY
```
