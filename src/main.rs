// See LICENSE file for copyright and license details.

#[macro_use]
extern crate glium;

extern crate clock_ticks;

use std::thread;
use std::time::Duration;
use glium::{glutin, DisplayBuild, Surface};
use glium::index::PrimitiveType;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

implement_vertex!(Vertex, position, color);

pub enum Action {
    Stop,
    Continue,
}

pub fn start_loop<F>(mut callback: F)
    where F: FnMut() -> Action
{
    let mut accumulator = 0;
    let mut previous_clock = clock_ticks::precise_time_ns();
    loop {
        match callback() {
            Action::Stop => break,
            Action::Continue => ()
        }
        let now = clock_ticks::precise_time_ns();
        accumulator += now - previous_clock;
        previous_clock = now;
        const FIXED_TIME_STAMP: u64 = 16666667;
        while accumulator >= FIXED_TIME_STAMP {
            accumulator -= FIXED_TIME_STAMP;
            // if you have a game, update the state here
        }
        thread::sleep(Duration::from_millis(
            ((FIXED_TIME_STAMP - accumulator) / 1000000) as u64));
    }
}

fn make_program(display: &glium::Display) -> glium::Program {
    let api = display.get_window().unwrap().get_api();
    let shader_src_preamble = match api {
        glium::glutin::Api::OpenGl => r"
            #version 120
            #define lowp
            #define mediump
            #define highp
        ",
        glium::glutin::Api::OpenGlEs => r"
            #version 100
        ",
        _ => unimplemented!(),
    }.to_string();
    println!("OZKRIFF: API: {:?}", api);
    let vertex_shader_src = shader_src_preamble.clone() + r"
        uniform lowp mat4 matrix;
        attribute lowp vec2 position;
        attribute lowp vec3 color;
        varying lowp vec3 vColor;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0) * matrix;
            vColor = color;
        }
    ";
    let fragment_shader_src = shader_src_preamble + r"
        varying lowp vec3 vColor;
        void main() {
            gl_FragColor = vec4(vColor, 1.0);
        }
    ";
    let program = glium::Program::from_source(
        display,
        &vertex_shader_src,
        &fragment_shader_src,
        None
    ).unwrap();
    program
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let gl_version = glutin::GlRequest::GlThenGles {
        opengles_version: (2, 0),
        opengl_version: (2, 0)
    };
    let display = glutin::WindowBuilder::new()
        .with_gl(gl_version)
        .with_title("Nergul".to_string())
        .build_glium()
        .unwrap();
    let vertex_buffer = {
        let vertices = [
            Vertex { position: [-0.5, -0.5], color: [0.0, 1.0, 0.0] },
            Vertex { position: [ 0.0,  0.5], color: [0.0, 0.0, 1.0] },
            Vertex { position: [ 0.5, -0.5], color: [1.0, 0.0, 0.0] },
        ];
        glium::VertexBuffer::new(&display, &vertices).unwrap()
    };
    let index_buffer = glium::IndexBuffer::new(
        &display, PrimitiveType::TrianglesList, &[0u16, 1, 2]).unwrap();
    let program = make_program(&display);
    start_loop(|| {
        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32],
            ]
        };
        {
            // fn draw() { ...
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 0.0);
            target.draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniforms,
                &Default::default(),
            ).unwrap();
            target.finish().unwrap();
        }
        for event in display.poll_events() {
            match event {
                glutin::Event::Closed => return Action::Stop,
                _ => ()
            }
        }
        Action::Continue
    });
}

// vim: set tabstop=4 shiftwidth=4 softtabstop=4 expandtab:
