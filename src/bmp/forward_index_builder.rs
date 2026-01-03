//! Streaming forward index builder that builds BlockForwardIndex directly.
//!
//! This avoids the intermediate ForwardIndex structure by building blocks
//! incrementally as documents are processed.

use bmp::index::forward_index::BlockForwardIndex;
use std::collections::HashMap;

/// Streaming builder for BlockForwardIndex.
///
/// Builds the block forward index directly without creating an intermediate
/// ForwardIndex structure. Processes one document at a time and flushes
/// completed blocks to reduce memory usage.
pub struct StreamingBlockForwardIndexBuilder {
    /// Block size
    bsize: usize,

    /// Number of documents
    num_documents: usize,

    /// Current block being built: term_id -> [(doc_offset, score)]
    current_block: HashMap<u16, Vec<(u8, u8)>>,

    /// Current block index
    current_block_idx: usize,

    /// Completed blocks
    completed_blocks: Vec<Vec<(u16, Vec<(u8, u8)>)>>,

    /// Last document ID processed (for validation)
    last_doc_id: Option<u32>,
}

impl StreamingBlockForwardIndexBuilder {
    /// Creates a new streaming block forward index builder.
    ///
    /// # Arguments
    /// * `num_documents` - Total number of documents
    /// * `bsize` - Block size for document grouping
    pub fn new(num_documents: usize, bsize: usize) -> Self {
        let num_blocks = (num_documents + bsize - 1) / bsize;
        Self {
            bsize,
            num_documents,
            current_block: HashMap::new(),
            current_block_idx: 0,
            completed_blocks: Vec::with_capacity(num_blocks),
            last_doc_id: None,
        }
    }

    /// Adds a document with its term-score pairs.
    ///
    /// Documents must be added in order (doc_id = 0, 1, 2, ...).
    ///
    /// # Arguments
    /// * `doc_id` - The document ID
    /// * `terms` - Iterator of (term_id, quantized_score) pairs
    pub fn add_document<I>(&mut self, doc_id: u32, terms: I)
    where
        I: IntoIterator<Item = (u16, u8)>,
    {
        // Validate document order
        if let Some(last) = self.last_doc_id {
            debug_assert!(
                doc_id > last,
                "Documents must be added in order: got {} after {}",
                doc_id,
                last
            );
        }
        self.last_doc_id = Some(doc_id);

        let block_idx = doc_id as usize / self.bsize;
        let doc_offset_in_block = (doc_id as usize % self.bsize) as u8;

        // Flush previous blocks if we've moved to a new one
        while block_idx > self.current_block_idx {
            self.flush_current_block();
        }

        // Add term-score pairs for this document
        for (term_id, score) in terms {
            self.current_block
                .entry(term_id)
                .or_insert_with(Vec::new)
                .push((doc_offset_in_block, score));
        }
    }

    /// Flushes the current block to completed_blocks.
    fn flush_current_block(&mut self) {
        if self.current_block.is_empty() {
            self.completed_blocks.push(Vec::new());
        } else {
            // Convert HashMap to sorted Vec
            let mut block_data: Vec<(u16, Vec<(u8, u8)>)> = self.current_block.drain().collect();
            block_data.sort_by_key(|(term_id, _)| *term_id);

            self.completed_blocks.push(block_data);
        }
        self.current_block_idx += 1;
    }

    /// Builds the final BlockForwardIndex.
    pub fn build(mut self) -> BlockForwardIndex {
        // Flush any remaining block
        if !self.current_block.is_empty() || self.current_block_idx < self.expected_num_blocks() {
            self.flush_current_block();
        }

        // Ensure we have the right number of blocks
        while self.completed_blocks.len() < self.expected_num_blocks() {
            self.completed_blocks.push(Vec::new());
        }

        BlockForwardIndex {
            data: self.completed_blocks,
            block_size: self.bsize,
        }
    }

    fn expected_num_blocks(&self) -> usize {
        (self.num_documents + self.bsize - 1) / self.bsize
    }
}

/// Alternative builder that processes postings term-by-term.
///
/// When the source index provides term-oriented iteration (like SparseIndex),
/// this builder accumulates postings and converts to block format.
pub struct TermOrientedBlockForwardIndexBuilder {
    /// Block size
    bsize: usize,

    /// Per-block data: block_idx -> term_id -> [(doc_offset, score)]
    blocks: Vec<HashMap<u16, Vec<(u8, u8)>>>,
}

impl TermOrientedBlockForwardIndexBuilder {
    /// Creates a new term-oriented builder.
    pub fn new(num_documents: usize, bsize: usize) -> Self {
        let num_blocks = (num_documents + bsize - 1) / bsize;
        let blocks = (0..num_blocks).map(|_| HashMap::new()).collect();

        Self { bsize, blocks }
    }

    /// Adds a posting for a term.
    ///
    /// # Arguments
    /// * `term_id` - The term ID (as u16)
    /// * `doc_id` - The document ID
    /// * `quantized_score` - The quantized impact score
    #[inline]
    pub fn add_posting(&mut self, term_id: u16, doc_id: u32, quantized_score: u8) {
        let block_idx = doc_id as usize / self.bsize;
        let doc_offset = (doc_id as usize % self.bsize) as u8;

        if block_idx < self.blocks.len() {
            self.blocks[block_idx]
                .entry(term_id)
                .or_insert_with(Vec::new)
                .push((doc_offset, quantized_score));
        }
    }

    /// Builds the final BlockForwardIndex.
    pub fn build(self) -> BlockForwardIndex {
        let data: Vec<Vec<(u16, Vec<(u8, u8)>)>> = self
            .blocks
            .into_iter()
            .map(|block_map| {
                let mut block_data: Vec<(u16, Vec<(u8, u8)>)> = block_map.into_iter().collect();
                block_data.sort_by_key(|(term_id, _)| *term_id);
                block_data
            })
            .collect();

        BlockForwardIndex {
            data,
            block_size: self.bsize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_builder_empty() {
        let builder = StreamingBlockForwardIndexBuilder::new(100, 32);
        let index = builder.build();

        assert_eq!(index.block_size, 32);
        assert_eq!(index.data.len(), 4); // ceil(100/32) = 4 blocks
    }

    #[test]
    fn test_streaming_builder_single_doc() {
        let mut builder = StreamingBlockForwardIndexBuilder::new(100, 32);

        builder.add_document(0, vec![(10, 50), (20, 60)]);

        let index = builder.build();

        assert_eq!(index.data.len(), 4);
        assert_eq!(index.data[0].len(), 2); // Two terms in first block

        // Check first block has the right terms
        let block0 = &index.data[0];
        assert_eq!(block0[0].0, 10); // term_id 10
        assert_eq!(block0[0].1, vec![(0, 50)]); // doc_offset 0, score 50
        assert_eq!(block0[1].0, 20); // term_id 20
        assert_eq!(block0[1].1, vec![(0, 60)]); // doc_offset 0, score 60
    }

    #[test]
    fn test_streaming_builder_multiple_blocks() {
        let mut builder = StreamingBlockForwardIndexBuilder::new(100, 32);

        // Doc in block 0
        builder.add_document(5, vec![(1, 10)]);
        // Doc in block 1
        builder.add_document(40, vec![(1, 20), (2, 30)]);
        // Doc in block 2
        builder.add_document(70, vec![(1, 40)]);

        let index = builder.build();

        assert_eq!(index.data.len(), 4);

        // Block 0: doc 5, term 1
        assert_eq!(index.data[0].len(), 1);
        assert_eq!(index.data[0][0].0, 1);
        assert_eq!(index.data[0][0].1, vec![(5, 10)]);

        // Block 1: doc 40, terms 1 and 2
        assert_eq!(index.data[1].len(), 2);
        assert_eq!(index.data[1][0].0, 1);
        assert_eq!(index.data[1][0].1, vec![(8, 20)]); // 40 % 32 = 8
        assert_eq!(index.data[1][1].0, 2);
        assert_eq!(index.data[1][1].1, vec![(8, 30)]);

        // Block 2: doc 70, term 1
        assert_eq!(index.data[2].len(), 1);
        assert_eq!(index.data[2][0].0, 1);
        assert_eq!(index.data[2][0].1, vec![(6, 40)]); // 70 % 32 = 6
    }

    #[test]
    fn test_term_oriented_builder() {
        let mut builder = TermOrientedBlockForwardIndexBuilder::new(100, 32);

        // Add postings term by term
        // Term 1 appears in docs 5, 40, 70
        builder.add_posting(1, 5, 10);
        builder.add_posting(1, 40, 20);
        builder.add_posting(1, 70, 40);

        // Term 2 appears in doc 40
        builder.add_posting(2, 40, 30);

        let index = builder.build();

        assert_eq!(index.data.len(), 4);

        // Same expected results as streaming builder test
        assert_eq!(index.data[0].len(), 1);
        assert_eq!(index.data[0][0].0, 1);
        assert_eq!(index.data[0][0].1, vec![(5, 10)]);
    }

    #[test]
    fn test_term_sorting_within_block() {
        let mut builder = TermOrientedBlockForwardIndexBuilder::new(32, 32);

        // Add terms out of order
        builder.add_posting(100, 0, 10);
        builder.add_posting(50, 0, 20);
        builder.add_posting(75, 0, 30);

        let index = builder.build();

        // Terms should be sorted within the block
        assert_eq!(index.data[0][0].0, 50);
        assert_eq!(index.data[0][1].0, 75);
        assert_eq!(index.data[0][2].0, 100);
    }
}
