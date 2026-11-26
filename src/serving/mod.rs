//! Online serving: retrieve candidates, rank them and return the top k for one user.

pub mod api;
pub mod bundle;
pub mod cache;
pub mod recommender;

#[cfg(test)]
mod tests;
