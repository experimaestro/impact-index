use std::{fs::File, ops::Deref};

use memmap2::{Mmap, MmapOptions};
use std::convert::AsRef;

pub trait Buffer: Send + AsRef<[u8]> + Deref<Target=[u8]> {}

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

impl Buffer for MmapBuffer {}

impl AsRef<[u8]> for MmapBuffer {
    fn as_ref(&self) -> &[u8] {
        todo!()
    }
}

impl Deref for MmapBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        todo!()
    }
    
}