extern crate enigo;
extern crate regex;


use enigo::*;
use enigo::MouseButton;
use std::fs::File;
//use std::fs::read_to_string;
use regex::Regex;
use std::io::Read;
use std::result::Result;
use std::error::Error;

// "centroid": [758, 121]
fn find_centroid(input: &str) -> (i32, i32) {
    let centroid_re = Regex::new("centroid\": \\[(\\d+), (\\d+)\\]" ).unwrap();
    let x= 0;
    let y= 0;
    for cap in centroid_re.captures_iter(input) {
        let x = String::from(&cap[1]).parse::<i32>().unwrap();
        let y = String::from(&cap[2]).parse::<i32>().unwrap();
        return (x, y)
    }
    panic!("didn't find anything")

}

fn move_mouse(enigo: &mut Enigo, x: i32, y: i32) {
    enigo.mouse_move_to(x, y);
    enigo.mouse_down(MouseButton::Left);
    println!("moved");
}


fn main() {
//    let mut enigo = Enigo::new();
//    enigo.mouse_move_to(500, 200);
//    enigo.mouse_down(MouseButton::Left);
//    enigo.mouse_move_relative(100, 100);
//    enigo.mouse_up(MouseButton::Left);
//    enigo.key_sequence("hello world");

    let mut input_file = File::open("out").expect("couldn't find input");
    let mut input_str = String::new();
    input_file.read_to_string(& mut input_str);
    let (x, y) = find_centroid(&input_str);

    let mut enigo = Enigo::new();
    enigo.mouse_move_to(x, y);
    enigo.mouse_down(MouseButton::Left);


}

