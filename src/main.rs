use micrograd_rust::engine::*;

fn main() {
    // Create inputs:
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    // Create weights:
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    // Create bias:
    let b = Value::new(6.8813735870195432);

    // Compute x1*w1 and x2*w2:
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();

    // Sum: x1*w1 + x2*w2
    let sum = x1w1 + x2w2;

    // Add bias: n = (x1*w1 + x2*w2) + b
    let n = sum + b.clone();

    // Apply tanh: o = tanh(n)

    let o = n.tanh();
    let o_clone = o.clone();
    o.backward(); // This should propagate gradients through the entire graph.

    // Finally, print the output value and the gradients of the inputs.
    println!("Output (o): {}", o_clone.0.borrow().data);
    println!("x1.grad: {}", x1.0.borrow().grad);
    println!("x2.grad: {}", x2.0.borrow().grad);
    println!("w1.grad: {}", w1.0.borrow().grad);
    println!("w2.grad: {}", w2.0.borrow().grad);
    println!("b.grad: {}", b.0.borrow().grad);
}
