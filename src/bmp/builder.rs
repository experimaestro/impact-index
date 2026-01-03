//! Main streaming BMP conversion builder.
//!
//! Provides memory-efficient conversion from SparseIndex to BMP format
//! using streaming builders that don't store raw postings in memory.

use std::fs::File;
use std::io::{BufWriter, Result};
use std::path::Path;

use bmp::index::forward_index::BlockForwardIndex;
use indicatif::{ProgressBar, ProgressStyle};

use crate::base::ImpactValue;
use crate::index::SparseIndexView;

use super::forward_index_builder::TermOrientedBlockForwardIndexBuilder;
use super::index::Index;
use super::posting_list_builder::StreamingPostingListManager;

const DEFAULT_PROGRESS_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})";

fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(DEFAULT_PROGRESS_TEMPLATE)
        .progress_chars("=> ")
}

/// Quantization levels for impact scores (8-bit quantization)
const LEVELS: i32 = 256;

/// Computes the global value range across all terms.
fn compute_value_range(index: &dyn SparseIndexView) -> (f32, f32) {
    let mut min_value = f32::INFINITY;
    let mut max_value = 0.0f32;

    for term_ix in 0..index.len() {
        let (_min, _max) = index.value_range(term_ix);
        max_value = max_value.max(_max);
        min_value = min_value.min(_min);
    }

    (min_value, max_value)
}

/// Creates a quantization function for impact scores.
fn make_quantizer(min_value: f32, max_value: f32) -> impl Fn(f32) -> u8 {
    let step = (LEVELS as f32) / (max_value - min_value);
    move |value: f32| (((value - min_value) * step) as i32).clamp(0, LEVELS - 1) as u8
}

/// Converts a SparseIndex to BMP format using streaming (memory-efficient) builders.
///
/// This is a two-pass algorithm that avoids storing raw postings in memory:
/// - Pass 1: Build posting lists (block max scores, k-th percentiles)
/// - Pass 2: Build block forward index
///
/// Memory usage is O(num_terms * num_blocks) instead of O(total_postings).
///
/// # Arguments
/// * `index` - The source sparse index
/// * `output` - Output path for the BMP index file
/// * `bsize` - Block size for BMP partitioning
/// * `compress_range` - Whether to compress block max scores
pub fn convert_to_bmp_streaming(
    index: &dyn SparseIndexView,
    output: &Path,
    bsize: usize,
    compress_range: bool,
) -> Result<()> {
    let num_terms = index.len();
    let num_documents = (index.max_doc_id() + 1) as usize;

    eprintln!(
        "Streaming BMP conversion: {} terms, {} documents, block size {}",
        num_terms, num_documents, bsize
    );

    // Compute quantization parameters
    let (min_value, max_value) = compute_value_range(index);
    let quantize = make_quantizer(min_value, max_value);

    // === Pass 1: Build posting lists ===
    eprintln!("Pass 1: Building posting lists (streaming)");
    let mut posting_manager = StreamingPostingListManager::new(num_terms, num_documents, bsize);

    let progress = ProgressBar::new(num_terms as u64);
    progress.set_style(pb_style());

    for term_ix in 0..num_terms {
        for posting in index.iterator(term_ix) {
            let quantized_score = quantize(posting.value);
            posting_manager.add_posting(term_ix, posting.docid as u32, quantized_score);
        }
        progress.inc(1);
    }
    progress.finish();

    let posting_lists = posting_manager.build(compress_range);
    eprintln!("  Built {} posting lists", posting_lists.len());

    // === Pass 2: Build block forward index ===
    eprintln!("Pass 2: Building block forward index (streaming)");
    let mut fwd_builder = TermOrientedBlockForwardIndexBuilder::new(num_documents, bsize);

    let progress = ProgressBar::new(num_terms as u64);
    progress.set_style(pb_style());

    for term_ix in 0..num_terms {
        let term_id = term_ix as u16;
        for posting in index.iterator(term_ix) {
            let quantized_score = quantize(posting.value);
            fwd_builder.add_posting(term_id, posting.docid as u32, quantized_score);
        }
        progress.inc(1);
    }
    progress.finish();

    let block_forward_index = fwd_builder.build();
    eprintln!("  Built {} blocks", block_forward_index.data.len());

    // === Build Index struct ===
    eprintln!("Building index structure");

    // Generate term names (using numeric IDs as strings, matching legacy behavior)
    let term_names: Vec<String> = (0..num_terms).map(|i| i.to_string()).collect();

    // Generate document names
    let documents: Vec<String> = (0..num_documents).map(|i| i.to_string()).collect();

    let inverted_index = Index::new(num_documents, posting_lists, term_names, documents);

    // === Serialize ===
    eprintln!("Serializing to {}", output.display());
    serialize_bmp(output, &inverted_index, &block_forward_index)?;

    // Print statistics
    print_statistics(&block_forward_index);

    Ok(())
}

/// Serializes the BMP index to a file.
fn serialize_bmp(
    output: &Path,
    inverted_index: &Index,
    block_forward_index: &BlockForwardIndex,
) -> Result<()> {
    let file = File::create(output)?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, &(inverted_index, block_forward_index))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(())
}

/// Prints statistics about the built index.
fn print_statistics(block_forward_index: &BlockForwardIndex) {
    let num_blocks = block_forward_index.data.len();
    if num_blocks == 0 {
        return;
    }

    let mut total_terms = 0;
    let mut total_avg_docs = 0.0;

    for block in &block_forward_index.data {
        total_terms += block.len();
        if !block.is_empty() {
            total_avg_docs +=
                block.iter().map(|(_, v)| v.len()).sum::<usize>() as f32 / block.len() as f32;
        }
    }

    eprintln!("Block statistics:");
    eprintln!("  Number of blocks: {}", num_blocks);
    eprintln!("  Avg terms per block: {}", total_terms / num_blocks);
    eprintln!(
        "  Avg docs per term (per block): {:.2}",
        total_avg_docs / num_blocks as f32
    );
}

/// Trait extension for SparseIndexView to provide value_range method.
pub trait SparseIndexViewExt {
    fn value_range(&self, term_ix: usize) -> (ImpactValue, ImpactValue);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer() {
        let quantize = make_quantizer(0.0, 1.0);

        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(1.0), 255);
        assert_eq!(quantize(0.5), 128);

        // Test clamping
        assert_eq!(quantize(-1.0), 0);
        assert_eq!(quantize(2.0), 255);
    }

    #[test]
    fn test_quantizer_with_offset() {
        let quantize = make_quantizer(10.0, 20.0);

        assert_eq!(quantize(10.0), 0);
        assert_eq!(quantize(20.0), 255);
        assert_eq!(quantize(15.0), 128);
    }
}
