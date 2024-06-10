use std::path::Path;

use super::index::SparseIndex;

/// Trait for all transforms
pub trait IndexTransform {
    /// Transforms the index
    fn process(self, path: &Path, index: &dyn SparseIndex) -> Result<(), std::io::Error>;
}
