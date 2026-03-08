# Python Documentation (Development)

This directory contains the Python type stubs and Sphinx documentation source for the `impact_index` module.

For the full user documentation, see: https://experimaestro-ir-rust.readthedocs.io/en/latest/index.html

## Files

- `impact_index.pyi` — Type stubs providing IDE autocompletion and type checking
- `docs/` — Sphinx documentation source

## Building the Documentation

```bash
cd python
make docs
```

Then open `docs/_build/index.html`.

## Updating `impact_index.pyi`

The `.pyi` stub file is **manually maintained**. It must be updated whenever the Python API changes in `src/py/mod.rs`.

### When to update

- A new `#[pyclass]` or `#[pymethods]` block is added
- A method signature changes (arguments, return type)
- A method or class is removed or renamed
- A `#[pyo3(signature = ...)]` default value changes

### How to update

1. Edit `src/py/mod.rs` with the Rust/PyO3 changes
2. Open `python/impact_index.pyi` and make the matching change:
   - New class → add a `class` block with docstring and typed methods
   - New method → add a method stub with type annotations and docstring
   - Changed signature → update the argument types / return type
   - Removed item → delete it from the stub
3. Rebuild docs to verify: `cd python && make docs`

### Type mapping reference

| Rust / PyO3 type          | Python stub type           |
|---------------------------|----------------------------|
| `bool`                    | `bool`                     |
| `usize`, `u64`, `i32`    | `int`                      |
| `f32`, `f64`              | `float`                    |
| `&str`, `String`          | `str`                      |
| `&[u8]`                   | `bytes`                    |
| `Vec<T>`                  | `list[T]`                  |
| `HashMap<K, V>`           | `dict[K, V]`               |
| `Option<T>`               | `Optional[T]`              |
| `(A, B)`                  | `tuple[A, B]`              |
| `&PyDict` (as input)      | `dict[K, V]`               |
| `PyObject` (return)       | Use the actual Python type (e.g., `list[ScoredDocument]`) |
| `&PyArray1<T>`            | `npt.NDArray[np.T]`        |
| `PyResult<&PyAny>` (async)| `Awaitable[T]`             |

### Future: automatic generation

When the project upgrades to PyO3 0.21+, `pyo3-stub-gen` (v0.17+) can auto-generate stubs. Until then, manual maintenance is required.
