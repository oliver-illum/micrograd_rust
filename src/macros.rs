#![allow(unused_imports)]
#![allow(unused_macros)]

use std::any::{Any, TypeId};

#[macro_export]
macro_rules! print_type {
    ($val:expr) => {{
        fn p<T>(_: &T) {
            println!("{}", std::any::type_name::<T>());
        }
        p(&$val);
    }};
}
macro_rules! is_type {
    ($val:expr, $t:ty) => {{
        TypeId::of::<$t>() == (&$val as &dyn Any).type_id()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_type_test() {
        let string = "string_test".to_string();
        let str_slice = "str_test";
        let integer = 42;

        //String
        assert!(is_type!(string, String));
        assert!(!is_type!(string, &str));

        //string literal
        assert!(is_type!(str_slice, &str));
        assert!(!is_type!(str_slice, String));

        //integer
        assert!(is_type!(integer, i32));
        assert!(!is_type!(integer, u32));
    }
}
