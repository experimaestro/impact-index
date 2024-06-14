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
    let module = PyModule::new(_py, "sparse")?;
    sparse::init(module)?;
    m.add_submodule(module)?;

    Ok(())
}
