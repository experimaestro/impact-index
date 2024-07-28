//! Methods for compressing the document IDs

use std::io::Write;

use super::{Compressor, DocIdCompressor, DocIdCompressorFactory, TermBlockInformation};
use crate::{
    base::{DocId, TermIndex},
    index::SparseIndexView,
    utils::buffer::Slice,
};
use ouroboros::self_referencing;
use serde::{Deserialize, Serialize};
use sucds::{EliasFano, EliasFanoBuilder, Searial};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct EliasFanoCompressor {}

#[typetag::serde]
impl DocIdCompressor for EliasFanoCompressor {}

impl DocIdCompressorFactory for EliasFanoCompressor {
    fn create(&self, _index: &dyn SparseIndexView) -> Box<dyn DocIdCompressor> {
        Box::new(EliasFanoCompressor {})
    }

    fn clone(&self) -> Box<dyn DocIdCompressorFactory> {
        Box::new(Clone::clone(self))
    }
}

#[self_referencing]
struct EliasFanoIterator {
    data: EliasFano,
    min_doc_id: DocId,
    #[borrows(data)]
    #[covariant]
    pub iterator: sucds::elias_fano::iter::Iter<'this>,
}

unsafe impl<'a> Send for EliasFanoIterator {}

impl<'a> Iterator for EliasFanoIterator {
    type Item = DocId;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_mut(|fields| {
            if let Some(x) = fields.iterator.next() {
                Some((x as DocId) + *fields.min_doc_id)
            } else {
                None
            }
        })
    }
}

impl Compressor<DocId> for EliasFanoCompressor {
    fn write(
        &self,
        writer: &mut dyn Write,
        values: &[DocId],
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) {
        let mut c = EliasFanoBuilder::new(
            (info.max_doc_id - info.min_doc_id + 1) as usize,
            values.len(),
        )
        .expect("Error when building");

        for &x in values {
            c.push((x - info.min_doc_id) as usize)
                .expect("Could not add a doc ID");
        }
        c.build()
            .serialize_into(writer)
            .expect("Error while serializing");
    }

    fn read<'a>(
        &self,
        slice: Box<dyn Slice + 'a>,
        _term_index: TermIndex,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = DocId> + Send + 'a> {
        let data = EliasFano::deserialize_from(slice.data()).expect("Error while reading");
        Box::new(
            EliasFanoIteratorBuilder {
                data: data,
                min_doc_id: info.min_doc_id,
                iterator_builder: |data: &EliasFano| data.iter(0),
            }
            .build(),
        )
    }
}
