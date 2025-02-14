#[allow(unused)]
use std::ops::{Add, Mul};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Parameter {
    pub data: f64,
    pub grad: f64, 
    //pub _op: String, 
    pub _prev: Vec<Parameter>,
    pub _backward: fn(&mut Parameter),
}

impl Parameter {
    pub fn new(data: f64) -> Self {
        Self {
            data, 
            grad: 0.0,
            //_op: "".into(),
            _prev: vec![],
            _backward: |_| {},
        }
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, 
            "Parameter(data={})",
            self.data
        )
    }
}