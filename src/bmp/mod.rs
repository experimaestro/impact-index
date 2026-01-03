//! Memory-efficient BMP (Block-Max Pruning) index building.
//!
//! This module provides streaming builders for creating BMP indices
//! without storing raw postings in memory. It may eventually replace
//! the BMP subcrate.
//!
//! ## Architecture
//!
//! - [`top_k_heap`]: Bounded heap for tracking top-K scores (used for k-th percentiles)
//! - [`posting_list_builder`]: Streaming posting list builder (block max scores)
//! - [`forward_index_builder`]: Streaming forward index builder
//! - [`index`]: Mirror struct for BMP's Index (for serialization)
//! - [`builder`]: Main conversion function
//!
//! ## Usage
//!
//! ```ignore
//! use impact_index::bmp::convert_to_bmp_streaming;
//!
//! // Convert a sparse index to BMP format
//! convert_to_bmp_streaming(&index, Path::new("output.bmp"), 64, true)?;
//! ```
//!
//! ## Memory Comparison
//!
//! | Component | Legacy | Streaming |
//! |-----------|--------|-----------|
//! | Raw postings | O(total_postings × 8B) | 0 |
//! | Block maxes | O(V × blocks) | O(V × blocks) |
//! | K-th scores | O(total_postings) sort | O(V × 1000) |
//! | Forward intermediate | O(total_postings × 8B) | O(blocks × terms_per_block) |

pub mod builder;
pub mod forward_index_builder;
pub mod index;
pub mod posting_list_builder;
pub mod top_k_heap;

// Re-export main conversion function
pub use builder::convert_to_bmp_streaming;

// Re-export key types
pub use forward_index_builder::{
    StreamingBlockForwardIndexBuilder, TermOrientedBlockForwardIndexBuilder,
};
pub use index::Index;
pub use posting_list_builder::{StreamingPostingListBuilder, StreamingPostingListManager};
pub use top_k_heap::TopKHeap;
