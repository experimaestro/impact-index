use std::path::Path;

use super::index::BlockTermImpactIndex;

/// Trait for all transforms
pub trait IndexTransform {
    /// Transforms the index
    fn process(self, path: &Path, index: &dyn BlockTermImpactIndex) -> Result<(), std::io::Error>;
}
