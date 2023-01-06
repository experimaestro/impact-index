//! Methods for compressing the posting lists

use std::{
    fs::File,
    io::{Seek, Read, Write},
    path::Path,
};

use super::index::BlockTermImpactIndex;
use crate::base::{DocId, ImpactValue};
use serde::{Serialize, Deserialize};
use sucds::{EliasFanoBuilder, Searial};

// 
// ---- Compression ---
// 
pub trait Compressor<T> {
    fn write(&self, writer: &mut dyn Write, values: &[T]);
    fn read(&self, reader: &mut dyn Read) -> Box<dyn Iterator<Item=T>>;
}

#[typetag::serde(tag = "type")]
pub trait DocIdCompressor : Compressor<usize> {}

#[typetag::serde(tag = "type")]
pub trait ValueCompressor : Compressor<ImpactValue> {}


// 
// ---- Compressed index global information  ---
// 

#[derive(Serialize, Deserialize)]
pub struct CompressedBlockInformation {
    /// Position for the document ID stream
    pub docid_position_range: (u64, u64),

    /// Position for the impact value stream
    pub value_position_range: (u64, u64),

    /// Number of records
    pub length: usize,

    /// Maximum value for this page
    pub max_value: ImpactValue,

    /// Maximum document ID for this page
    pub max_doc_id: DocId,
}

#[derive(Serialize, Deserialize)]
pub struct CompressedTermIndexInformation {
    pub pages: Vec<CompressedBlockInformation>,
    pub max_value: ImpactValue,
    pub max_doc_id: DocId,
    pub length: usize,
}


/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct CompressedIndexInformation {
    pub terms: Vec<CompressedTermIndexInformation>,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ValueCompressor>,
}

impl BlockTermImpactIndex for CompressedIndexInformation {
    fn iterator(&self, term_ix: crate::base::TermIndex) -> Box<dyn super::index::BlockTermImpactIterator + '_> {
        // TODO: Iterate over compressed index information
        todo!()
    }

    fn length(&self) -> usize {
        self.terms.len()
    }
}



/// Compress the impact values
/// 
/// # Arguments
/// 
/// - max_block_size: maximum number of records per block
pub fn compress(
    path: &Path,
    index: &dyn BlockTermImpactIndex,
    max_block_size: usize,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ValueCompressor>,
) -> Result<(), std::io::Error> {
    // File for impact values
    let mut value_writer = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path.join("value.dat"))
        .expect("Could not create the values file");

    // File for document IDs
    let mut docid_writer = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path.join("docid.dat"))
        .expect("Could not create the document IDs file");

    // Global information
    let mut information = CompressedIndexInformation { 
        terms: Vec::new(),
        doc_ids_compressor: doc_ids_compressor,
        values_compressor: values_compressor
    };
    let value_position = 0;
    let docid_position = 0;
    
    // Iterate over terms
    for term_ix in 0..index.length() {
        // Read everything
        let mut it = index.iterator(term_ix);
        let mut flag = true;

        let mut term_information = CompressedTermIndexInformation {
            pages: Vec::new(),
            max_value: 0f32,
            max_doc_id: 0,
            length: 0,
        };
        
        while flag {
            // Read up to max_block_size records
            let mut impacts = Vec::new();
            let mut docids = Vec::<usize>::new();
            flag = false;
            while let Some(ti) = it.next() {
                docids.push(ti.docid as usize);
                impacts.push(ti.value);
                if docids.len() == max_block_size {
                    flag = true;
                    break;
                }
            }   

            // Write
            information.doc_ids_compressor.write(&mut value_writer, &docids);
            information.values_compressor.write(&mut docid_writer, &impacts);

            // Add information
            let new_value_position = value_writer.stream_position()?;
            let new_docid_position = docid_writer.stream_position()?;
            let block_term_information = CompressedBlockInformation {
                docid_position_range: (docid_position, new_docid_position),
                value_position_range: (value_position, new_value_position),
                length: impacts.len(),
                max_value: impacts.iter().fold(0f32, |cur, x| { cur.max(*x) }),
                max_doc_id: docids.iter().fold(0 as usize, |cur, x| -> usize { cur.max(*x) }).try_into().unwrap()
            };

            term_information.max_value = term_information.max_value.max(block_term_information.max_value);
            term_information.max_doc_id = term_information.max_doc_id.max(block_term_information.max_doc_id);
            term_information.length += block_term_information.length;
            term_information.pages.push(block_term_information);
        }

        information.terms.push(term_information);
    }

    // Serialize the overall structure
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .write(true)
        .truncate(true)
        .open(info_path)
        .expect("Error while creating file");

    ciborium::ser::into_writer(&information, info_file).expect("Error saving compressed term index information");

    Ok(())
}

pub fn load_compressed_index(path: &Path) -> CompressedIndexInformation {
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .read(true)
        .open(info_path)
        .expect("Error while creating file");

    ciborium::de::from_reader(info_file).expect("Error loading compressed term index information")
}

// --- Elias Fano

struct EliasFanoCompressor {
}

impl<'a> Compressor<usize> for EliasFanoCompressor {
    fn write(&self, writer: &mut dyn Write, values: &[usize]) {
        let max_value = *values.iter().max().unwrap();

        let mut c = EliasFanoBuilder::new(max_value, values.len())
                .expect("Error when building");

        c.append(values)
            .expect("Could not add a doc ID");
        c.build().serialize_into(writer).expect("Yoooo");
    }

    fn read(&self, reader: &mut dyn Read) -> Box<dyn Iterator<Item=usize>> {
        todo!()
    }
}
