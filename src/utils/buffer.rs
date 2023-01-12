use std::fs::File;

use memmap2::{Mmap, MmapOptions};

pub struct Slice<'a> {
    pub data: &'a [u8],
}

pub trait Buffer: Send + Sync {
    fn slice(&'_ self, start: usize, end: usize) -> Slice<'_>;
}

pub struct MemoryBuffer {}

pub struct MmapBuffer {
    mmap: Mmap,
}

impl MmapBuffer {
    pub fn new(file: &File) -> Self {
        let mmap = unsafe {
            MmapOptions::new()
                .map(file)
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
