from __future__ import annotations

import pytest
import tempfile

from pathlib import Path

import triton

from triton.runtime import build
from triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


@pytest.mark.parametrize(
    ("system", "language", "expected"),
    (
        ("Darwin", "c", "/usr/bin/clang"),
        ("Darwin", "c++", "/usr/bin/clang++"),
        ("Linux", "c", "/usr/bin/gcc"),
        ("Linux", "c++", "/usr/bin/g++"),
    ),
)
def test_find_compiler_uses_platform_default(monkeypatch, system, language, expected):
    compilers = {
        "clang": "/usr/bin/clang",
        "gcc": "/usr/bin/gcc",
        "clang++": "/usr/bin/clang++",
        "g++": "/usr/bin/g++",
    }
    monkeypatch.setattr(build.platform, "system", lambda: system)
    monkeypatch.setattr(build.shutil, "which", compilers.get)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    build._find_compiler.cache_clear()
    try:
        assert build._find_compiler(language) == expected
    finally:
        build._find_compiler.cache_clear()
