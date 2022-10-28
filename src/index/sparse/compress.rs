//! Methods for compressing the posting lists


use std::{io::{Write, Seek}, fs::File, path::Path};

use sucds::{EliasFanoBuilder, Searial};
use crate::base::{ImpactValue, DocId};
use super::{wand::{WandIndex, WandIterator}};


pub trait Compressor<T> {
    fn add(&mut self, value: &T);
}

pub trait CompressorFactory<T> {
    fn init<'a>(&self, value_writer: &'a mut dyn Write, it: &dyn WandIterator) -> Box<dyn Compressor<T> + 'a>;
}

struct EliasFanoCompressor<'a> {
    builder: EliasFanoBuilder,
    writer: &'a mut dyn Write
}
impl<'a> Compressor<DocId> for EliasFanoCompressor<'a> {
    fn add(&mut self, value: &DocId) {
        self.builder.append(&[*value as usize]).expect("Could not add a doc ID");
    }
}
impl<'a> Drop for EliasFanoCompressor<'a> {
    fn drop(&mut self) {
        let mut other = EliasFanoBuilder::new(1,1).expect("Yaaa");
        std::mem::swap(&mut other, &mut self.builder);
        other.build().serialize_into(&mut self.writer).expect("Yoooo");
    }
}

struct EliasFanoCompressorFactory {
}
impl CompressorFactory<DocId> for EliasFanoCompressorFactory {
    fn init<'a>(&self, writer: &'a mut dyn Write, it: &dyn WandIterator) -> Box<dyn Compressor<DocId> + 'a> {
        Box::new(EliasFanoCompressor {
            builder: EliasFanoBuilder::new(it.max_doc_id().try_into().unwrap(), it.length()).expect("Error when building"),
            writer: writer
        })
    }
}

/// Compress the impact values
pub fn compress(index: &dyn WandIndex, 
    doc_compressor_factory: &dyn CompressorFactory<DocId>,
    value_compressor_factory: &dyn CompressorFactory<ImpactValue>, 
) -> Result<(), std::io::Error> {

    let path = Path::new("/tmp/yaya");
    let mut value_writer = File::options().write(true).truncate(true).create(true).open(path).expect("Could not create the data file");
    let mut docid_writer = File::options().write(true).truncate(true).create(true).open(path).expect("Could not create the data file");


    for term_ix in 0..index.length() {
        let mut it = index.iterator(term_ix);
        {
            let mut doc_compressor = doc_compressor_factory.init(&mut value_writer, it.as_ref());
            let mut value_compressor = value_compressor_factory.init(&mut docid_writer, it.as_ref());
            
            while let Some(ti) = it.next() {
                doc_compressor.add(&ti.docid);
                value_compressor.add(&ti.value);
            }
        }
        value_writer.stream_position()?;
    }

    Ok(())
}