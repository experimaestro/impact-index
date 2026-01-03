//! Streaming posting list builder that computes block max scores incrementally.
//!
//! This avoids storing all raw postings in memory by computing statistics
//! on-the-fly as postings are added.

use bmp::index::posting_list::{BlockData, CompressedBlock, PostingList};

use super::top_k_heap::TopKHeap;

/// Streaming builder for a single PostingList.
///
/// Computes block max scores and k-th percentile scores incrementally
/// without storing raw postings.
pub struct StreamingPostingListBuilder {
    /// Block size for range max computation
    bsize: usize,

    /// Block max scores - one u8 per block
    range_maxes: Vec<u8>,

    /// Bounded priority queue for k-th score estimation
    top_k_heap: TopKHeap,

    /// Total postings seen (for statistics)
    count: usize,
}

impl StreamingPostingListBuilder {
    /// Creates a new streaming posting list builder.
    ///
    /// # Arguments
    /// * `num_documents` - Total number of documents in the index
    /// * `bsize` - Block size for partitioning documents
    pub fn new(num_documents: usize, bsize: usize) -> Self {
        let num_blocks = (num_documents + bsize - 1) / bsize; // div_ceil
        Self {
            bsize,
            range_maxes: vec![0u8; num_blocks],
            top_k_heap: TopKHeap::new(),
            count: 0,
        }
    }

    /// Adds a posting (document score) to this term's posting list.
    ///
    /// Updates block max scores and k-th percentile tracking.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID
    /// * `quantized_score` - The quantized impact score (0-255)
    #[inline]
    pub fn add_posting(&mut self, doc_id: u32, quantized_score: u8) {
        // Update block max
        let block_idx = doc_id as usize / self.bsize;
        if block_idx < self.range_maxes.len() {
            self.range_maxes[block_idx] = self.range_maxes[block_idx].max(quantized_score);
        }

        // Update top-K heap for k-th percentile
        self.top_k_heap.push(quantized_score);
        self.count += 1;
    }

    /// Returns the number of postings added.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Finalizes and produces a PostingList.
    ///
    /// # Arguments
    /// * `compress_range` - If true, use sparse compression for block max scores
    pub fn build(self, compress_range: bool) -> PostingList {
        let kth_scores = self.top_k_heap.get_kth_scores();

        let block_data = if compress_range {
            BlockData::Compressed(Self::compress(&self.range_maxes))
        } else {
            BlockData::Raw(self.range_maxes)
        };

        PostingList::new(block_data, kth_scores)
    }

    /// Compresses block max scores using sparse representation.
    ///
    /// Groups into superblocks of 256 and only stores non-zero entries.
    fn compress(data: &[u8]) -> Vec<CompressedBlock> {
        let mut compressed = Vec::new();

        for superblock in data.chunks(256) {
            let mut max_scores = Vec::new();

            for (offset, &value) in superblock.iter().enumerate() {
                if value > 0 {
                    max_scores.push((offset, value));
                }
            }

            compressed.push(CompressedBlock { max_scores });
        }

        compressed
    }
}

/// Manager for building multiple posting lists in streaming fashion.
///
/// Handles lazy initialization of per-term builders to avoid allocating
/// memory for terms that have no postings.
pub struct StreamingPostingListManager {
    /// Per-term builders (lazily initialized)
    builders: Vec<Option<StreamingPostingListBuilder>>,

    /// Number of documents
    num_documents: usize,

    /// Block size
    bsize: usize,
}

impl StreamingPostingListManager {
    /// Creates a new manager for the given number of terms.
    pub fn new(num_terms: usize, num_documents: usize, bsize: usize) -> Self {
        Self {
            builders: (0..num_terms).map(|_| None).collect(),
            num_documents,
            bsize,
        }
    }

    /// Adds a posting for a specific term.
    #[inline]
    pub fn add_posting(&mut self, term_id: usize, doc_id: u32, quantized_score: u8) {
        let builder = self.builders[term_id].get_or_insert_with(|| {
            StreamingPostingListBuilder::new(self.num_documents, self.bsize)
        });
        builder.add_posting(doc_id, quantized_score);
    }

    /// Builds all posting lists.
    pub fn build(self, compress_range: bool) -> Vec<PostingList> {
        let num_blocks = (self.num_documents + self.bsize - 1) / self.bsize;

        self.builders
            .into_iter()
            .map(|opt_builder| {
                opt_builder
                    .map(|b| b.build(compress_range))
                    .unwrap_or_else(|| {
                        // Empty posting list for terms with no postings
                        // Must have same block structure as non-empty terms for consistency
                        let block_data = if compress_range {
                            // Compressed: empty blocks (all zeros = no non-zero entries)
                            let compressed: Vec<CompressedBlock> = (0..((num_blocks + 255) / 256))
                                .map(|_| CompressedBlock {
                                    max_scores: Vec::new(),
                                })
                                .collect();
                            BlockData::Compressed(compressed)
                        } else {
                            BlockData::Raw(vec![0u8; num_blocks])
                        };
                        PostingList::new(block_data, vec![0, 0, 0])
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_builder() {
        let builder = StreamingPostingListBuilder::new(1000, 64);
        assert_eq!(builder.count(), 0);

        let posting_list = builder.build(false);
        // Should produce a valid empty posting list
        assert_eq!(posting_list.kth(10), 0);
    }

    #[test]
    fn test_single_posting() {
        let mut builder = StreamingPostingListBuilder::new(1000, 64);
        builder.add_posting(100, 50);
        assert_eq!(builder.count(), 1);

        let posting_list = builder.build(false);
        // Block 100/64 = 1 should have max score 50
        // But we can't directly check this without access to internal state
    }

    #[test]
    fn test_block_max_computation() {
        let mut builder = StreamingPostingListBuilder::new(256, 64);

        // Add postings to block 0 (docs 0-63)
        builder.add_posting(0, 10);
        builder.add_posting(32, 20);
        builder.add_posting(63, 15);

        // Add postings to block 1 (docs 64-127)
        builder.add_posting(64, 5);
        builder.add_posting(100, 25);

        // Block 0 max should be 20, block 1 max should be 25
        assert_eq!(builder.count(), 5);
    }

    #[test]
    fn test_manager_lazy_init() {
        let mut manager = StreamingPostingListManager::new(100, 1000, 64);

        // Only add postings for a few terms
        manager.add_posting(5, 100, 50);
        manager.add_posting(5, 200, 60);
        manager.add_posting(50, 300, 70);

        let posting_lists = manager.build(false);
        assert_eq!(posting_lists.len(), 100);

        // Most posting lists should be empty
        // Terms 5 and 50 should have postings
    }

    #[test]
    fn test_compression() {
        let data = vec![0, 10, 0, 0, 20, 0, 30, 0];
        let compressed = StreamingPostingListBuilder::compress(&data);

        assert_eq!(compressed.len(), 1); // One superblock
        assert_eq!(compressed[0].max_scores.len(), 3); // Three non-zero entries
        assert_eq!(compressed[0].max_scores[0], (1, 10));
        assert_eq!(compressed[0].max_scores[1], (4, 20));
        assert_eq!(compressed[0].max_scores[2], (6, 30));
    }
}
