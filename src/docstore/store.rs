use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use memmap2::{Mmap, MmapOptions};

use crate::base::BoxResult;

use super::{
    key_fst_file, BlockMeta, Document, DocumentStoreMeta, BLOCKS_FILE, CONTENT_FILE, META_FILE,
    OFFSETS_FILE,
};

/// How content.dat is accessed.
#[derive(Debug, Clone, Copy)]
pub enum ContentAccess {
    /// Load entire file into RAM (fastest, high RAM).
    Memory,
    /// Memory-mapped (OS manages paging, low RAM).
    Mmap,
    /// Read blocks from disk on demand via seek+read (lowest RAM).
    Disk,
}

enum ContentBackend {
    Memory(Vec<u8>),
    Mmap(Mmap),
    Disk(Mutex<File>),
}

/// Read-only document store with block-based zstd compression.
pub struct DocumentStore {
    meta: DocumentStoreMeta,
    content: ContentBackend,
    blocks_data: Vec<u8>,
    offsets_data: Vec<u8>,
    key_fsts: HashMap<String, fst::Map<Vec<u8>>>,
}

impl DocumentStore {
    /// Load a document store from disk.
    pub fn load(dir: &Path, access: ContentAccess) -> BoxResult<Self> {
        let meta_file = File::open(dir.join(META_FILE))?;
        let meta: DocumentStoreMeta = ciborium::de::from_reader(meta_file)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;

        let content_path = dir.join(CONTENT_FILE);
        let content = match access {
            ContentAccess::Memory => {
                let mut f = File::open(&content_path)?;
                let mut data = Vec::new();
                f.read_to_end(&mut data)?;
                ContentBackend::Memory(data)
            }
            ContentAccess::Mmap => {
                let f = File::open(&content_path)?;
                let mmap = unsafe { MmapOptions::new().map(&f)? };
                ContentBackend::Mmap(mmap)
            }
            ContentAccess::Disk => {
                let f = File::open(&content_path)?;
                ContentBackend::Disk(Mutex::new(f))
            }
        };

        let mut blocks_data = Vec::new();
        File::open(dir.join(BLOCKS_FILE))?.read_to_end(&mut blocks_data)?;

        let mut offsets_data = Vec::new();
        File::open(dir.join(OFFSETS_FILE))?.read_to_end(&mut offsets_data)?;

        let mut key_fsts = HashMap::new();
        for key_name in &meta.key_names {
            let fst_path = dir.join(key_fst_file(key_name));
            let mut fst_data = Vec::new();
            File::open(&fst_path)?.read_to_end(&mut fst_data)?;
            let map = fst::Map::new(fst_data)
                .map_err(|e| format!("Failed to load FST for key '{}': {}", key_name, e))?;
            key_fsts.insert(key_name.clone(), map);
        }

        Ok(Self {
            meta,
            content,
            blocks_data,
            offsets_data,
            key_fsts,
        })
    }

    /// Number of documents in the store.
    pub fn num_documents(&self) -> u64 {
        self.meta.num_documents
    }

    /// Available key names.
    pub fn key_names(&self) -> &[String] {
        &self.meta.key_names
    }

    #[inline]
    fn get_block_index(&self, doc_number: u64) -> u32 {
        let offset = (doc_number as usize) * 4;
        u32::from_le_bytes(self.offsets_data[offset..offset + 4].try_into().unwrap())
    }

    #[inline]
    fn get_intra_offset(&self, doc_number: u64) -> u32 {
        let n = self.meta.num_documents as usize;
        let offset = n * 4 + (doc_number as usize) * 4;
        u32::from_le_bytes(self.offsets_data[offset..offset + 4].try_into().unwrap())
    }

    #[inline]
    fn get_block_meta(&self, block_index: u32) -> BlockMeta {
        let offset = (block_index as usize) * BlockMeta::SIZE;
        BlockMeta::from_bytes(&self.blocks_data[offset..])
    }

    fn decompress_block(&self, block_meta: &BlockMeta) -> BoxResult<Vec<u8>> {
        let offset = block_meta.offset as usize;
        let size = block_meta.compressed_size as usize;

        match &self.content {
            ContentBackend::Memory(data) => Ok(zstd::decode_all(&data[offset..offset + size])?),
            ContentBackend::Mmap(mmap) => Ok(zstd::decode_all(&mmap[offset..offset + size])?),
            ContentBackend::Disk(file_mutex) => {
                let mut file = file_mutex
                    .lock()
                    .map_err(|e| format!("Lock error: {}", e))?;
                let mut buf = vec![0u8; size];
                file.seek(SeekFrom::Start(block_meta.offset))?;
                file.read_exact(&mut buf)?;
                Ok(zstd::decode_all(&buf[..])?)
            }
        }
    }

    #[inline]
    fn decode_document_at(
        decompressed: &[u8],
        intra_offset: u32,
    ) -> BoxResult<(HashMap<String, String>, Vec<u8>)> {
        let mut pos = intra_offset as usize;

        let keys_len = u64::from_le_bytes(decompressed[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let keys: HashMap<String, String> =
            bincode::deserialize(&decompressed[pos..pos + keys_len])?;
        pos += keys_len;

        let content_len =
            u64::from_le_bytes(decompressed[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let content = decompressed[pos..pos + content_len].to_vec();

        Ok((keys, content))
    }

    /// Retrieve documents by their sequential numbers (0-based).
    pub fn get_by_number(&self, doc_numbers: &[u64]) -> BoxResult<Vec<Document>> {
        for &num in doc_numbers {
            if num >= self.meta.num_documents {
                return Err(format!(
                    "Document number {} out of range (num_documents={})",
                    num, self.meta.num_documents
                )
                .into());
            }
        }

        // Fast path for single document retrieval (avoids HashMap overhead)
        if doc_numbers.len() == 1 {
            let doc_num = doc_numbers[0];
            let block_index = self.get_block_index(doc_num);
            let intra_offset = self.get_intra_offset(doc_num);
            let block_meta = self.get_block_meta(block_index);
            let decompressed = self.decompress_block(&block_meta)?;
            let (keys, content) = Self::decode_document_at(&decompressed, intra_offset)?;
            return Ok(vec![Document {
                internal_id: doc_num,
                keys,
                content,
            }]);
        }

        // Group by block index to minimize decompression
        let mut block_groups: HashMap<u32, Vec<(usize, u64, u32)>> = HashMap::new();
        for (result_idx, &doc_num) in doc_numbers.iter().enumerate() {
            let block_index = self.get_block_index(doc_num);
            let intra_offset = self.get_intra_offset(doc_num);
            block_groups
                .entry(block_index)
                .or_default()
                .push((result_idx, doc_num, intra_offset));
        }

        let mut results: Vec<Option<Document>> = vec![None; doc_numbers.len()];

        for (block_index, docs_in_block) in &block_groups {
            let block_meta = self.get_block_meta(*block_index);
            let decompressed = self.decompress_block(&block_meta)?;

            for &(result_idx, doc_num, intra_offset) in docs_in_block {
                let (keys, content) = Self::decode_document_at(&decompressed, intra_offset)?;
                results[result_idx] = Some(Document {
                    internal_id: doc_num,
                    keys,
                    content,
                });
            }
        }

        Ok(results.into_iter().map(|d| d.unwrap()).collect())
    }

    /// Retrieve documents by key name and values.
    /// Returns one Option<Document> per input value (None if not found).
    pub fn get_by_key(
        &self,
        key_name: &str,
        key_values: &[&str],
    ) -> BoxResult<Vec<Option<Document>>> {
        let fst = self
            .key_fsts
            .get(key_name)
            .ok_or_else(|| format!("Unknown key name '{}'", key_name))?;

        let mut found_indices: Vec<(usize, u64)> = Vec::new();

        for (i, &val) in key_values.iter().enumerate() {
            if let Some(doc_num) = fst.get(val) {
                found_indices.push((i, doc_num));
            }
        }

        if found_indices.is_empty() {
            return Ok(vec![None; key_values.len()]);
        }

        let doc_nums: Vec<u64> = found_indices.iter().map(|&(_, num)| num).collect();
        let mut docs = self.get_by_number(&doc_nums)?;

        let mut results: Vec<Option<Document>> = vec![None; key_values.len()];
        // Iterate in reverse to pop from the end (avoids shifting)
        for (found_pos, &(input_idx, _)) in found_indices.iter().enumerate().rev() {
            let doc = std::mem::replace(
                &mut docs[found_pos],
                Document {
                    internal_id: 0,
                    keys: HashMap::new(),
                    content: Vec::new(),
                },
            );
            results[input_idx] = Some(doc);
        }

        Ok(results)
    }
}
