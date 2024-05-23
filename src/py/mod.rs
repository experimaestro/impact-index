use log::debug;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod sparse;

/// A Python module implemented in Rust.
#[pymodule]
fn xpmir_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Init logging
    pyo3_log::init();
    debug!("Loading xpmir-rust extension");

    // Index submodule
    let submod = PyModule::new(_py, "index")?;
    m.add_submodule(submod)?;

    submod.add_class::<sparse::PySparseIndexer>()?;
    submod.add_class::<sparse::PySparseBuilderIndex>()?;

    Ok(())
}
