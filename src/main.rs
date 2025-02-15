use micrograd_rust::print_type;

fn main() {
    println!("Hello, world!");

    let nn = String::from("stringstring");
    print_type!(nn);
}
