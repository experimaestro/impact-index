pub type TermIndex = usize;
pub type ImpactValue = f32;
pub type DocId = u64;
pub type BoxResult<T> = Result<T,Box<dyn std::error::Error>>;
