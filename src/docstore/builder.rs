use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::base::BoxResult;

use super::{
    BlockMeta, Document, DocumentStoreMeta, BLOCKS_FILE, CONTENT_FILE, META_FILE, OFFSETS_FILE,
};

/// Builds a DocumentStore on disk with block-based zstd compression.
pub struct DocumentStoreBuilder {
    dir: PathBuf,
    block_size: usize,
    zstd_level: i32,
    current_block_buf: Vec<u8>,
    current_block_doc_count: u32,
    doc_block_indices: Vec<u32>,
    doc_intra_offsets: Vec<u32>,
    /// Temp files for each key: lines of "key_value\tdoc_num\n"
    key_files: HashMap<String, BufWriter<File>>,
    key_names: Vec<String>,
    data_file: BufWriter<File>,
    blocks_file: BufWriter<File>,
    num_blocks: usize,
    num_documents: u64,
    content_offset: u64,
}

impl DocumentStoreBuilder {
    /// Create a new builder writing to the given directory.
    pub fn new(dir: &Path, block_size: usize, zstd_level: i32) -> BoxResult<Self> {
        fs::create_dir_all(dir)?;

        let data_file = BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(CONTENT_FILE))?,
        );
        let blocks_file = BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(BLOCKS_FILE))?,
        );

        Ok(Self {
            dir: dir.to_path_buf(),
            block_size,
            zstd_level,
            current_block_buf: Vec::new(),
            current_block_doc_count: 0,
            doc_block_indices: Vec::new(),
            doc_intra_offsets: Vec::new(),
            key_files: HashMap::new(),
            key_names: Vec::new(),
            data_file,
            blocks_file,
            num_blocks: 0,
            num_documents: 0,
            content_offset: 0,
        })
    }

    /// Add a document to the store.
    pub fn add(&mut self, doc: &Document) -> BoxResult<()> {
        let block_index = self.num_blocks as u32;
        let intra_offset = self.current_block_buf.len() as u32;

        self.doc_block_indices.push(block_index);
        self.doc_intra_offsets.push(intra_offset);

        // Serialize into current_block_buf:
        // keys (bincode) + content_len (u64 LE) + content bytes
        let keys_bytes = bincode::serialize(&doc.keys)?;
        let keys_len = keys_bytes.len() as u64;
        self.current_block_buf
            .extend_from_slice(&keys_len.to_le_bytes());
        self.current_block_buf.extend_from_slice(&keys_bytes);

        let content_len = doc.content.len() as u64;
        self.current_block_buf
            .extend_from_slice(&content_len.to_le_bytes());
        self.current_block_buf.extend_from_slice(&doc.content);

        // Write key entries to temp files
        let doc_num = self.num_documents;
        for (key_name, key_value) in &doc.keys {
            if !self.key_files.contains_key(key_name) {
                let temp_path = self.dir.join(format!(".tmp_key_{}", key_name));
                let file = BufWriter::new(
                    File::options()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&temp_path)?,
                );
                self.key_files.insert(key_name.clone(), file);
                self.key_names.push(key_name.clone());
            }
            let writer = self.key_files.get_mut(key_name).unwrap();
            writeln!(writer, "{}\t{}", key_value, doc_num)?;
        }

        self.num_documents += 1;
        self.current_block_doc_count += 1;

        if self.current_block_buf.len() >= self.block_size {
            self.flush_block()?;
        }

        Ok(())
    }

    fn flush_block(&mut self) -> BoxResult<()> {
        if self.current_block_buf.is_empty() {
            return Ok(());
        }

        let compressed = zstd::encode_all(&self.current_block_buf[..], self.zstd_level)?;

        self.data_file.write_all(&compressed)?;

        let block_meta = BlockMeta {
            offset: self.content_offset,
            compressed_size: compressed.len() as u64,
            num_docs: self.current_block_doc_count,
        };
        self.blocks_file.write_all(&block_meta.to_bytes())?;

        self.content_offset += compressed.len() as u64;
        self.num_blocks += 1;
        self.current_block_buf.clear();
        self.current_block_doc_count = 0;

        Ok(())
    }

    /// Finalize and write all index files.
    pub fn build(mut self) -> BoxResult<()> {
        // Flush remaining block
        self.flush_block()?;

        // Flush data and blocks files
        self.data_file.flush()?;
        self.blocks_file.flush()?;

        // Write offsets.dat: block_indices then intra_offsets as contiguous u32 LE arrays
        {
            let mut offsets_file = BufWriter::new(
                File::options()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(self.dir.join(OFFSETS_FILE))?,
            );
            for &idx in &self.doc_block_indices {
                offsets_file.write_all(&idx.to_le_bytes())?;
            }
            for &off in &self.doc_intra_offsets {
                offsets_file.write_all(&off.to_le_bytes())?;
            }
            offsets_file.flush()?;
        }

        // Build FSTs for each key
        // Close temp files first
        let key_names = self.key_names.clone();
        for name in &key_names {
            self.key_files.remove(name); // drops and closes
        }

        for key_name in &key_names {
            let temp_path = self.dir.join(format!(".tmp_key_{}", key_name));

            // Read all entries
            let reader = BufReader::new(File::open(&temp_path)?);
            let mut entries: Vec<(String, u64)> = Vec::new();
            for line in reader.lines() {
                let line = line?;
                let mut parts = line.splitn(2, '\t');
                let key_value = parts
                    .next()
                    .ok_or("invalid temp key file format")?
                    .to_string();
                let doc_num: u64 = parts
                    .next()
                    .ok_or("invalid temp key file format")?
                    .parse()?;
                entries.push((key_value, doc_num));
            }

            // Sort by key value
            entries.sort_by(|a, b| a.0.cmp(&b.0));

            // Check for duplicates and build FST
            let fst_path = self.dir.join(super::key_fst_file(key_name));
            let fst_file = BufWriter::new(
                File::options()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&fst_path)?,
            );
            let mut fst_builder = fst::MapBuilder::new(fst_file)?;

            for i in 0..entries.len() {
                if i > 0 && entries[i].0 == entries[i - 1].0 {
                    return Err(format!(
                        "Duplicate key value '{}' for key '{}'",
                        entries[i].0, key_name
                    )
                    .into());
                }
                fst_builder.insert(&entries[i].0, entries[i].1)?;
            }
            fst_builder.finish()?;

            // Clean up temp file
            fs::remove_file(&temp_path)?;
        }

        // Write metadata
        let meta = DocumentStoreMeta {
            num_documents: self.num_documents,
            block_size: self.block_size,
            num_blocks: self.num_blocks,
            key_names,
        };
        let meta_file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(self.dir.join(META_FILE))?;
        ciborium::ser::into_writer(&meta, meta_file)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        Ok(())
    }
}
