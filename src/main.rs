use micrograd_rust::engine::*;

fn main() {
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
