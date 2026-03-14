use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = impact_index::py::stub_info()?;
    stub.generate()?;
    Ok(())
}
