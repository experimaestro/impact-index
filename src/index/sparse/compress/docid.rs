//! Methods for compressing the document IDs

use std::io::Write;

use super::{Compressor, DocIdCompressor, TermBlockInformation};
use crate::{base::DocId, utils::buffer::Slice};
use ouroboros::self_referencing;
use serde::{Deserialize, Serialize};
use sucds::{EliasFano, EliasFanoBuilder, Searial};

#[derive(Serialize, Deserialize)]
pub struct EliasFanoCompressor {}

#[typetag::serde]
impl DocIdCompressor for EliasFanoCompressor {
    fn clone(&self) -> Box<dyn DocIdCompressor> {
        Box::new(EliasFanoCompressor {})
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
    fn write(&self, writer: &mut dyn Write, values: &[DocId], info: &TermBlockInformation) {
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
