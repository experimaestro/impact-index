
#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use crate::index::sparse;
    use ndarray::array;

    #[test]
    fn test_add() {
        let mut indexer = sparse::Indexer::new();

        let mut termix = array![3, 10, 25, 40];
        let mut values = array![1.2, 2.5, 3.1, 2.3];
        

        indexer.add(0, termix, values)
        assert_eq!(3, 3);
    }

}
