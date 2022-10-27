use pyo3::{types::PyModule, Python, PyResult, pymodule};

mod sparse;


/// A Python module implemented in Rust.
#[pymodule]
fn xpmir_rust(_py: Python, m: &PyModule) -> PyResult<()> { 
    // Index submodule
    let submod = PyModule::new(_py, "index")?;
    m.add_submodule(submod)?;

    submod.add_class::<sparse::PySparseIndexer>()?;
    submod.add_class::<sparse::PySparseBuilderIndex>()?;
    
    Ok(())
}