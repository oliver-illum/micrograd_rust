use crate::engine::{Value, ValueWrapper};
use rand::Rng;

pub struct Neuron {
    pub w: Vec<ValueWrapper>,
    pub b: ValueWrapper,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        Neuron {
            w: (0..nin)
                .map(|_| Value::new(rng.random_range(-1.0..1.0)))
                .collect(),
            b: Value::new(rng.random_range(-1.0..1.0)),
        }
    }

    pub fn call(&self, x: &[ValueWrapper]) -> ValueWrapper {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = act + (wi.clone() * xi.clone());
        }
        act.tanh()
    }

    pub fn parameters(&self) -> Vec<ValueWrapper> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }

    pub fn call(&self, x: &[ValueWrapper]) -> Vec<ValueWrapper> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }

    pub fn parameters(&self) -> Vec<ValueWrapper> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MultiLayerPerceptron {
    pub layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut input_size = nin;
        for &nout in nouts {
            layers.push(Layer::new(input_size, nout));
            input_size = nout;
        }

        MultiLayerPerceptron { layers }
    }

    pub fn call(&self, mut x: Vec<ValueWrapper>) -> Vec<ValueWrapper> {
        for layer in &self.layers {
            x = layer.call(&x)
        }
        x
    }

    pub fn parameters(&self) -> Vec<ValueWrapper> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
