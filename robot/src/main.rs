extern crate enigo;
extern crate regex;

use std::thread::sleep;
use enigo::*;
use enigo::MouseButton;
use std::fs::File;
//use std::fs::read_to_string;
use regex::Regex;
use std::io::Read;
use std::result::Result;
use std::error::Error;
use std::{thread, time};

// "centroid": [758, 121]
fn find_centroids(input: &str) -> Vec<(i32, i32)>  {
    let centroid_re = Regex::new("centroid\": \\[(\\d+), (\\d+)\\]" ).unwrap();
    let x= 0;
    let y= 0;
    let mut centroids: Vec<(i32, i32)> = Vec::new();
    for cap in centroid_re.captures_iter(input) {
        let x = String::from(&cap[1]).parse::<i32>().unwrap();
        let y = String::from(&cap[2]).parse::<i32>().unwrap();
        centroids.push((x, y))
    }
    return centroids;

}

fn move_mouse(enigo: &mut Enigo, x: i32, y: i32) {
    enigo.mouse_move_to(x, y);
    enigo.mouse_down(MouseButton::Left);
    println!("moved");
}


fn main() {
    let mut input_file = File::open("out2").expect("couldn't find input");
    let mut input_str = String::new();
    input_file.read_to_string(& mut input_str);


    let centroids = find_centroids(&input_str);
    for i in (0..centroids.len()) {
          let second = time::Duration::from_secs(1);
          thread::sleep(second);
          let (x, y) = centroids[i];
          println!("centroid, {}, {}", x, y);
          let mut enigo = Enigo::new();
          enigo.mouse_move_to(x, y);
    }

}

