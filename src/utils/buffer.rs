use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use memmap2::{Mmap, MmapOptions};

pub struct Slice<'a> {
    pub data: &'a [u8],
}

pub trait Buffer: Send + Sync {
    fn slice(&'_ self, start: usize, end: usize) -> Slice<'_>;
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
    fn slice(&'_ self, start: usize, end: usize) -> Slice<'_> {
        Slice {
            data: &self.data[start..end],
        }
    }
}

/// Uses a memory map
pub struct MmapBuffer {
    mmap: Mmap,
}

impl MmapBuffer {
    pub fn new(path: &PathBuf) -> Self {
        let mut file = File::options()
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

impl Buffer for MmapBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Slice<'_> {
        Slice {
            data: &self.mmap[start..end],
        }
    }
}
