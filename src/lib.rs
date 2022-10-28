#[macro_use]
extern crate simple_error;


pub mod search;
pub mod base;
pub mod index {
    pub mod sparse;
}

mod py;

#[cfg(test)]
mod tests;