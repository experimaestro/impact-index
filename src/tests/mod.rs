
#[cfg(test)]
mod tests {
    use std::path::Path;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use crate::index::sparse::{self, ForwardIndexTrait};
    use ndarray::array;

    #[test]
    fn test_add() {
        let mut indexer = sparse::Indexer::new(Path::new("/tmp/yo"));

        let termix = array![3, 10, 25, 40];
        let values = array![1.2, 2.5, 3.1, 2.3];
        

        let r = indexer.add(0, &termix, &values);
        r.expect("Error while adding terms to the index");

        for ti in indexer.iter(0) {
            println!("{}: {}", ti.docid, ti.value)
        }
        assert_eq!(3, 3);
    }

}
