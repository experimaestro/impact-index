use std::path::Path;

pub mod split;
use super::index::SparseIndexView;

/// Trait for all transforms
pub trait IndexTransform: Send + Sync {
    /// Transforms the index
    fn process(&self, path: &Path, index: &dyn SparseIndexView) -> Result<(), std::io::Error>;
}
