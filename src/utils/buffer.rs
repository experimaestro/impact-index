use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

pub trait Slice: Send + Sync {
    fn data(&'_ self) -> &'_ [u8];
}

struct MemorySlice<'a> {
    _data: &'a [u8],
}

impl Slice for MemorySlice<'_> {
    fn data(&'_ self) -> &'_ [u8] {
        self._data
    }
}

pub trait Buffer: Send + Sync {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice + '_>;
}

/// Stores the data in memory
pub struct MemoryBuffer {
    data: Vec<u8>,
}

impl MemoryBuffer {
    pub fn new(path: &PathBuf) -> Self {
        let mut file = File::options()
            .read(true)
            .open(path)
            .expect("Error while reading posting file");

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .expect("Error while reading file");

        Self { data: data }
    }
}

impl Buffer for MemoryBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice + '_> {
        Box::new(MemorySlice {
            _data: &self.data[start..end],
        })
    }
}

/// Uses a memory map
pub struct MmapBuffer {
    mmap: Mmap,
}

impl MmapBuffer {
    pub fn new(path: &PathBuf) -> Self {
        let file = File::options()
            .read(true)
            .open(path)
            .expect("Error while reading posting file");
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .expect("Cannot create a memory map")
        };
        Self { mmap: mmap }
    }
}

struct MmapSlice {
    vector: Vec<u8>,
}

impl Slice for MmapSlice {
    fn data(&'_ self) -> &'_ [u8] {
        &self.vector
    }
}

impl Buffer for MmapBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice> {
        let vector = Vec::from_iter(self.mmap[start..end].iter().map(|t| *t));
        let _data: &[u8] = &vector;

        Box::new(MmapSlice { vector: vector })
    }
}
