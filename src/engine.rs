//use crate::is_type;

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
#[allow(unused)]
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub _op: String,
    pub _prev: Vec<ValueWrapper>,
    pub _backward: Option<fn(&mut ValueWrapper)>,
}

impl Value {
    pub fn new(data: f64) -> ValueWrapper {
        ValueWrapper(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            _op: "".to_string(),
            _prev: vec![],
            _backward: None,
        })))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write!(f, "Value(data={})", self.data)
        write!(
            f,
            "Value(data={},grad={},operand={},prev_num={})",
            self.data,
            self.grad,
            self._op,
            self._prev.len()
        )
    }
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("_prev", &self._prev)
            .field("_backward", &"Box<dyn FnMut()>")
            .finish()
    }
}

#[derive(Clone)]
pub struct ValueWrapper(pub Rc<RefCell<Value>>);

impl ValueWrapper {
    pub fn pow(self, exponent: f64) -> ValueWrapper {
        let out_data = self.0.borrow().data.powf(exponent);
        let out = Value::new(out_data);
        {
            let mut out_borrow = out.0.borrow_mut();
            out_borrow._op = format!("**{}", exponent);
            out_borrow._prev = vec![self.clone()];
            fn _backward(out: &mut ValueWrapper) {
                let op_str = out.0.borrow()._op.clone();
                let exponent: f64 = op_str.trim_start_matches("**").parse().unwrap_or(1.0);
                let grad = out.0.borrow().grad;
                let parent = &out.0.borrow()._prev[0];
                let x = parent.0.borrow().data;
                parent.0.borrow_mut().grad += exponent * x.powf(exponent - 1.0) * grad;
            }
            out_borrow._backward = Some(_backward);
        }
        out
    }
    pub fn exp(self) -> ValueWrapper {
        let out_data = self.0.borrow().data.exp();
        let out = Value::new(out_data);
        {
            let mut out_borrow = out.0.borrow_mut();
            out_borrow._op = "exp".to_string();
            out_borrow._prev = vec![self.clone()];
            fn _backward(out: &mut ValueWrapper) {
                let grad = out.0.borrow().grad;
                let parent = &out.0.borrow()._prev[0];
                parent.0.borrow_mut().grad += out.0.borrow().data * grad;
            }
            out_borrow._backward = Some(_backward);
        }
        out
    }

    pub fn tanh(self) -> ValueWrapper {
        let x = self.0.borrow().data;
        let t = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp());
        let out = Value::new(t);
        {
            let mut out_borrow = out.0.borrow_mut();
            out_borrow._op = "tanh".to_string();
            out_borrow._prev = vec![self.clone()];
            fn _backward(out: &mut ValueWrapper) {
                let grad = out.0.borrow().grad;
                let parent = &out.0.borrow()._prev[0];
                let t_val = out.0.borrow().data;
                parent.0.borrow_mut().grad += (1.0 - t_val * t_val) * grad;
            }
            out_borrow._backward = Some(_backward);
        }
        out
    }

    pub fn backward(self) {
        fn build_topo(
            node: &ValueWrapper,
            visited: &mut HashSet<ValueWrapper>,
            topo: &mut Vec<ValueWrapper>,
        ) {
            if visited.contains(node) {
                return;
            }
            visited.insert(node.clone());
            let children = {
                let node_ref = node.0.borrow();
                node_ref._prev.clone()
            };
            for child in children {
                build_topo(&child, visited, topo);
            }
            topo.push(node.clone());
        }
        fn topological_list(start: &ValueWrapper) -> Vec<ValueWrapper> {
            let mut visited = HashSet::new();
            let mut topo = Vec::new();
            build_topo(start, &mut visited, &mut topo);
            topo
        }
        self.0.borrow_mut().grad = 1.0;

        let topo = topological_list(&self);
        for node in topo.into_iter().rev() {
            if let Some(backward_fn) = node.0.borrow()._backward {
                backward_fn(&mut node.clone());
            }
        }
    }
}

impl fmt::Debug for ValueWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.borrow())
    }
}

// Implement conversion from f64 to our ValueWrapper.
// This lets us write, for example: let a: ValueWrapper = 2.0.into(); or let c = a * 2.0
impl From<f64> for ValueWrapper {
    fn from(value: f64) -> Self {
        Value::new(value)
    }
}

impl PartialEq for ValueWrapper {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ValueWrapper {}

impl Hash for ValueWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state)
    }
}

impl<T> Add<T> for ValueWrapper
where
    T: Into<ValueWrapper>,
{
    type Output = ValueWrapper;

    fn add(self, other: T) -> ValueWrapper {
        let other: ValueWrapper = other.into();
        let out_data = self.0.borrow().data + other.0.borrow().data;
        let out = Value::new(out_data);

        {
            let mut out_borrow = out.0.borrow_mut();
            out_borrow._op = "+".to_string();
            out_borrow._prev = vec![self.clone(), other.clone()];
            fn _backward(out: &mut ValueWrapper) {
                let grad = out.0.borrow().grad;
                for prev in out.0.borrow()._prev.iter() {
                    prev.0.borrow_mut().grad += grad;
                }
            }
            out_borrow._backward = Some(_backward);
        }

        out
    }
}

impl<T> Mul<T> for ValueWrapper
where
    T: Into<ValueWrapper>,
{
    type Output = ValueWrapper;

    fn mul(self, other: T) -> ValueWrapper {
        let other = other.into();
        let out_data = self.0.borrow().data * other.0.borrow().data;
        let out = Value::new(out_data);
        {
            let mut out_borrow = out.0.borrow_mut();
            out_borrow._op = "*".to_string();
            out_borrow._prev = vec![self.clone(), other.clone()];

            fn _backward(out: &mut ValueWrapper) {
                let grad = out.0.borrow().grad;
                let left = &out.0.borrow()._prev[0];
                let right = &out.0.borrow()._prev[1];
                left.0.borrow_mut().grad += right.0.borrow().data * grad;
                right.0.borrow_mut().grad += left.0.borrow().data * grad;
            }
            out_borrow._backward = Some(_backward);
        }
        out
    }
}
impl Neg for ValueWrapper {
    type Output = ValueWrapper;

    fn neg(self) -> ValueWrapper {
        self * -1.0
    }
}

impl<T> Sub<T> for ValueWrapper
where
    T: Into<ValueWrapper>,
{
    type Output = ValueWrapper;

    fn sub(self, other: T) -> ValueWrapper {
        self + (-other.into()) // Just use `self + (-other)`
    }
}

impl<T> Div<T> for ValueWrapper
where
    T: Into<ValueWrapper>,
{
    type Output = ValueWrapper;

    fn div(self, other: T) -> ValueWrapper {
        self * (other.into().pow(-1.0)) // Uses power for division
    }
}
