use micrograd_rust::{engine::*, nn::*};

fn main() {
    let xs: Vec<Vec<ValueWrapper>> = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];

    let ys: Vec<ValueWrapper> = vec![1.0.into(), (-1.0).into(), (-1.0).into(), 1.0.into()];

    let mlp = MultiLayerPerceptron::new(3, &[4, 1]);

    println!("First predictions:");
    for (i, x) in xs.iter().enumerate() {
        let output = mlp.call(x.clone());
        let prediction = output[0].0.borrow().data;
        println!("Sample {}: Prediction = {}", i, prediction);
    }
    println!();
    for epoch in 0..=100 {
        let mut ypreds: Vec<ValueWrapper> = Vec::new();
        for x in &xs {
            let output = mlp.call(x.clone());
            ypreds.push(output[0].clone());
        }

        let mut loss = Value::new(0.0);
        for (ygt, ypred) in ys.iter().zip(ypreds.iter()) {
            let error = ypred.clone() - ygt.clone();
            let error_sq = error.pow(2.0);
            loss = loss + error_sq;
        }

        for p in mlp.parameters() {
            p.0.borrow_mut().grad = 0.0;
        }

        let loss_clone = loss.clone();
        loss.backward();

        for p in mlp.parameters() {
            let mut param = p.0.borrow_mut();
            param.data += -0.1 * param.grad;
        }
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss_clone.0.borrow().data);
        }
    }
    println!();
    println!("Final predictions:");
    for (i, x) in xs.iter().enumerate() {
        let output = mlp.call(x.clone());
        let prediction = output[0].0.borrow().data;
        println!("Sample {}: Prediction = {}", i, prediction);
    }
}

#[allow(unused)]
fn initialize_mpl() {
    let mlp = MultiLayerPerceptron::new(3, &[4, 1]);
    let x = vec![Value::new(1.0), Value::new(2.0), Value::new(3.0)];

    let output = mlp.call(x);
    println!("Output: {:?}", output);
    println!("Prediction: {}", output[0].0.borrow().data);
}

#[allow(unused)]
fn test_value() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.8813735870195432);

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();

    let sum = x1w1 + x2w2;

    let n = sum + b.clone();

    let o = n.tanh();
    let o_clone = o.clone();
    o.backward(); // This should propagate gradients through the entire graph.

    println!("Output (o): {}", o_clone.0.borrow().data);
    println!("x1.grad: {}", x1.0.borrow().grad);
    println!("x2.grad: {}", x2.0.borrow().grad);
    println!("w1.grad: {}", w1.0.borrow().grad);
    println!("w2.grad: {}", w2.0.borrow().grad);
    println!("b.grad: {}", b.0.borrow().grad);
}
