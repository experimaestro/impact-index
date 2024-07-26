use helpers::index::{check_same_index, TestIndex};
use impact_index::index::SparseIndex;
use impact_index::transforms::split::SplitIndexTransform;
use impact_index::{
    base::load_index,
    compress::{
        docid::EliasFanoCompressor,
        impact::{Identity, Quantizer},
        CompressionTransform,
    },
    transforms::IndexTransform,
};
use temp_dir::TempDir;

#[test]
fn test_split_index() {
    let mut data = TestIndex::new(100, 1000, 5., 10, None, Some(10));
    let index = data.indexer.to_index(true);
    let dir = TempDir::new().expect("Could not create temporary directory");

    let sink = Box::new(CompressionTransform {
        max_block_size: 1024,
        doc_ids_compressor: Box::new(EliasFanoCompressor {}),
        impacts_compressor: Box::new(Identity {}),
    });

    let transform = SplitIndexTransform {
        sink,
        quantiles: [1. / 64.].to_vec(),
    };

    transform
        .process(dir.path(), &index)
        .expect("An error occurred");

    let c_index = load_index(dir.path(), true);

    check_same_index(
        c_index.block_iterator(0).as_mut(),
        index.block_iterator(0).as_mut(),
        0.,
    );
}
