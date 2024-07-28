use helpers::index::{check_same_index, TestIndex};
use impact_index::index::SparseIndex;
use impact_index::{
    base::load_index,
    compress::{
        docid::EliasFanoCompressor, impact::Identity, impact::Quantizer, CompressionTransform,
    },
    transforms::IndexTransform,
};
use rstest::rstest;
use temp_dir::TempDir;

#[rstest]
#[case(5. / ((2 << 4) as f64), CompressionTransform {
    max_block_size: 1024,
    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
    impacts_compressor_factory: Box::new(Quantizer::new(4, 0., 5.)),
})]
#[case(0., CompressionTransform {
    max_block_size: 1024,
    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
    impacts_compressor_factory: Box::new(Identity {})
})]
fn test_compressed_index(#[case] step: f64, #[case] transform: CompressionTransform) {
    let mut data = TestIndex::new(100, 1000, 5., 10, None, Some(10));
    let index = data.indexer.to_index(true);

    let dir = TempDir::new().expect("Could not create temporary directory");

    transform
        .process(dir.path(), &index)
        .expect("An error occurred");

    let c_index = load_index(dir.path(), true);

    check_same_index(
        c_index.block_iterator(0).as_mut(),
        index.block_iterator(0).as_mut(),
        step,
    );
}
