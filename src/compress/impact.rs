//! Methods for compressing impact values

use std::io::Write;

use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};
use byteorder::{ReadBytesExt, WriteBytesExt};

use super::{Compressor, ImpactCompressor, TermBlockInformation};
use crate::{
    base::ImpactValue,
    utils::buffer::{Slice, SliceReader},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]

pub struct Quantizer {
    pub nbits: u32,
    pub levels: u32,
    pub step: ImpactValue,
    pub min: ImpactValue,
    pub max: ImpactValue,
}

impl Quantizer {
    pub fn new(nbits: u32, min: ImpactValue, max: ImpactValue) -> Self {
        let levels = 2 << (nbits - 1);
        Self {
            nbits: nbits,
            levels: levels,
            min: min,
            max: max,
            step: (max - min) / ((levels + 1) as f32),
        }
    }
}

struct QuantizerIterator<'a> {
    nbits: u32,
    index: usize,
    count: usize,
    min: ImpactValue,
    step: ImpactValue,
    bit_reader: BitReader<Box<SliceReader<'a>>, bitstream_io::BigEndian>,
}

impl<'a> Iterator for QuantizerIterator<'a> {
    type Item = ImpactValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            self.index += 1;
            let quantized = self.bit_reader.read::<u32>(self.nbits).unwrap();
            let x = (quantized as ImpactValue) * self.step + self.min + self.step / 2.;
            Some(x)
        } else {
            None
        }
    }
}

#[typetag::serde]
impl ImpactCompressor for Quantizer {
    fn clone(&self) -> Box<dyn ImpactCompressor> {
        Box::new(Clone::clone(self))
    }
}

impl<'a> Compressor<ImpactValue> for Quantizer {
    fn write(&self, writer: &mut dyn Write, values: &[ImpactValue], _info: &TermBlockInformation) {
        let mut bit_writer = BitWriter::endian(writer, BigEndian);

        for x in values {
            let value = ((*x - self.min) / self.step).trunc() as u32;

            let quantized = value.max(0).min(self.levels - 1);
            bit_writer
                .write(self.nbits, quantized)
                .expect("Cannot write bits");
        }

        bit_writer
            .byte_align()
            .expect("Could not write padding bits");
    }

    fn read<'b>(
        &self,
        slice: Box<dyn Slice + 'b>,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'b> {
        let slice_reader = Box::new(SliceReader::new(slice));
        let bit_reader = BitReader::endian(slice_reader, BigEndian);

        Box::new(QuantizerIterator::<'b> {
            nbits: self.nbits,
            index: 0,
            count: info.length,
            bit_reader: bit_reader,
            min: self.min,
            step: self.step,
        })
    }
}

// ----- Identity transform

#[derive(Serialize, Deserialize, Clone)]
pub struct Identity {}

#[typetag::serde]
impl ImpactCompressor for Identity {
    fn clone(&self) -> Box<dyn ImpactCompressor> {
        Box::new(Clone::clone(self))
    }
}

impl<'a> Compressor<ImpactValue> for Identity {
    fn write(&self, writer: &mut dyn Write, values: &[ImpactValue], _info: &TermBlockInformation) {
        for x in values {
            writer
                .write_f32::<byteorder::BigEndian>(*x)
                .expect("cannot write");
        }
    }

    fn read<'b>(
        &self,
        slice: Box<dyn Slice + 'b>,
        info: &TermBlockInformation,
    ) -> Box<dyn Iterator<Item = ImpactValue> + Send + 'b> {
        Box::new(IdentityIterator::<'b> {
            index: 0,
            count: info.length,
            slice,
        })
    }
}

struct IdentityIterator<'a> {
    index: usize,
    count: usize,
    slice: Box<dyn Slice + 'a>,
}

impl<'a> Iterator for IdentityIterator<'a> {
    type Item = ImpactValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            let data = self.slice.as_ref().data();
            let mut view = &data[self.index * 4..self.index * 4 + 4];
            self.index += 1;
            Some(view.read_f32::<byteorder::BigEndian>().expect("read error"))
        } else {
            None
        }
    }
}
