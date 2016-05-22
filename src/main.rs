// See LICENSE file for copyright and license details.

#[macro_use]
extern crate glium;

extern crate time;
extern crate image;
extern crate cgmath;

#[cfg(target_os = "android")]
extern crate android_glue;

mod fs;

use std::path::{Path};
use std::f32::consts::{PI};
use std::thread;
use std::time::Duration;
use glium::{glutin, Texture2d, DisplayBuild, Surface};
use glium::index::PrimitiveType;
use glium::glutin::ElementState::{Pressed, Released};
use cgmath::{Matrix4, Matrix3, Vector3, Vector2, Rad};

const FPS: u64 = 60;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

implement_vertex!(Vertex, position, tex_coords);

fn load_texture<P: AsRef<Path>>(display: &glium::Display, path: P) -> Texture2d {
    let f = fs::load(path);
    let image = image::load(f, image::PNG).unwrap().to_rgba();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba(
        image.into_raw(), image_dimensions);
    Texture2d::new(display, image).unwrap()
}

fn make_program(display: &glium::Display) -> glium::Program {
    let api = display.get_window().unwrap().get_api();
    let pre_src = fs::load_string(match api {
        glutin::Api::OpenGl => "pre_gl.glsl",
        glutin::Api::OpenGlEs => "pre_gles.glsl",
        _ => unimplemented!(),
    });
    let vs_src = pre_src.clone() + &fs::load_string("vs.glsl");
    let fs_src = pre_src + &fs::load_string("fs.glsl");
    glium::Program::from_source(display, &vs_src, &fs_src, None).unwrap()
}

fn view_matrix(angle_x: Rad<f32>, angle_y: Rad<f32>, zoom: f32, aspect: f32) -> Matrix4<f32> {
    let perspective_mat = cgmath::perspective(Rad::new(PI / 4.0), aspect, 0.1, 100.0);
    let tr_mat = Matrix4::from_translation(Vector3{x: 0.0, y: 0.0, z: -zoom});
    let angle_x_m = Matrix4::from(Matrix3::from_angle_z(angle_x));
    let angle_y_m = Matrix4::from(Matrix3::from_angle_x(angle_y));
    perspective_mat * tr_mat * angle_y_m * angle_x_m
}

fn win_size(display: &glium::Display) -> (u32, u32) {
    let window = display.get_window().unwrap();
    window.get_inner_size().unwrap()
}

fn aspect(display: &glium::Display) -> f32 {
    let (x, y) = win_size(display);
    x as f32 / y as f32
}

fn create_display() -> glium::Display {
    let gl_version = glutin::GlRequest::GlThenGles {
        opengles_version: (2, 0),
        opengl_version: (2, 1),
    };
    glutin::WindowBuilder::new()
        .with_gl(gl_version)
        .with_depth_buffer(24)
        .with_title("Nergal".to_string())
        .build_glium()
        .unwrap()
}

struct Visualizer {
    display: glium::Display,
    program: glium::Program,
    is_running: bool,
    accumulator: u64,
    previous_clock: u64,
    vertex_buffer: glium::VertexBuffer<Vertex>,
    index_buffer: glium::IndexBuffer<u16>,
    texture: Texture2d,
    camera_angle_x: Rad<f32>,
    camera_angle_y: Rad<f32>,
    zoom: f32,
    aspect: f32,
    mouse_pos: Vector2<i32>,
    is_lmb_pressed: bool,
}

impl Visualizer {
    fn new() -> Visualizer {
        let display = create_display();
        let program = make_program(&display);
        let aspect = aspect(&display);
        let vertex_buffer = {
            let vertices = [
                Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 1.0] },
                Vertex { position: [-0.5,  0.5, 0.0], tex_coords: [0.0, 0.0] },
                Vertex { position: [ 0.5, -0.5, 0.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [ 0.5,  0.5, 0.0], tex_coords: [1.0, 0.0] },
            ];
            glium::VertexBuffer::new(&display, &vertices).unwrap()
        };
        let index_buffer = glium::IndexBuffer::new(
            &display, PrimitiveType::TrianglesList, &[0u16, 1, 2, 1, 2, 3]).unwrap();
        let texture = load_texture(&display, "test.png");
        let accumulator = 0;
        let previous_clock = time::precise_time_ns();
        Visualizer {
            display: display,
            program: program,
            is_running: true,
            accumulator: accumulator,
            previous_clock: previous_clock,
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            texture: texture,
            camera_angle_x: Rad::new(0.0),
            camera_angle_y: Rad::new(0.0),
            zoom: 5.0,
            aspect: aspect,
            mouse_pos: Vector2{x: 0, y: 0},
            is_lmb_pressed: false,
        }
    }

    fn is_running(&self) -> bool {
        self.is_running
    }

    fn handle_events(&mut self) {
        self.aspect = aspect(&self.display);
        let rotate_step = PI / 12.0;
        let events: Vec<_> = self.display.poll_events().collect();
        for event in events {
            match event {
                glutin::Event::Resized(x, y) => {
                    self.aspect = x as f32 / y as f32;
                },
                glutin::Event::Closed => {
                    self.is_running = false;
                },
                glutin::Event::KeyboardInput(Released, _, Some(key)) => {
                    match key {
                        glutin::VirtualKeyCode::Q
                            | glutin::VirtualKeyCode::Escape =>
                        {
                            self.is_running = false;
                        },
                        glutin::VirtualKeyCode::Right => self.camera_angle_x.s += rotate_step,
                        glutin::VirtualKeyCode::Left => self.camera_angle_x.s -= rotate_step,
                        glutin::VirtualKeyCode::Down => self.camera_angle_y.s += rotate_step,
                        glutin::VirtualKeyCode::Up => self.camera_angle_y.s -= rotate_step,
                        glutin::VirtualKeyCode::Equals => self.zoom *= 0.75,
                        glutin::VirtualKeyCode::Subtract => self.zoom *= 1.25,
                        _ => {},
                    }
                },
                glutin::Event::MouseMoved(x, y) => {
                    self.handle_mouse_move(Vector2{x: x, y: y});
                },
                glutin::Event::MouseInput(Pressed, glutin::MouseButton::Left) => {
                    self.is_lmb_pressed = true;
                },
                glutin::Event::MouseInput(Released, glutin::MouseButton::Left) => {
                    self.is_lmb_pressed = false;
                },
                glutin::Event::Touch(glutin::Touch{location: (x, y), phase, ..}) => {
                    let pos = Vector2{x: x as i32, y: y as i32};
                    match phase {
                        glutin::TouchPhase::Moved => {
                            self.handle_mouse_move(pos);
                        },
                        glutin::TouchPhase::Started => {
                            self.is_lmb_pressed = true;
                            self.mouse_pos = pos;
                        },
                        glutin::TouchPhase::Ended => {
                            self.mouse_pos = pos;
                            self.is_lmb_pressed = false;
                        },
                        glutin::TouchPhase::Cancelled => unimplemented!(),
                    }
                },
                _ => (),
            }
        }
    }

    fn handle_mouse_move(&mut self, pos: Vector2<i32>) {
        let diff = self.mouse_pos - pos;
        self.mouse_pos = pos;
        if self.is_lmb_pressed {
            let (w, h) = win_size(&self.display);
            self.camera_angle_x.s += PI * (diff.x as f32 / w as f32);
            self.camera_angle_y.s += PI * (diff.y as f32 / h as f32);
        }
    }

    fn draw(&self) {
        let model_pos = Vector3{x: 1.0, y: 0.0, z: 0.0};
        let model_mat: [[f32; 4]; 4] = Matrix4::from_translation(model_pos).into();
        let view_mat: [[f32; 4]; 4] = view_matrix(
            self.camera_angle_x, self.camera_angle_y, self.zoom, self.aspect).into();
        let uniforms = uniform! {
            view_mat: view_mat,
            model_mat: model_mat,
            texture: &self.texture,
        };
        let mut target = self.display.draw();
        target.clear_color(0.0, 0.0, 0.0, 0.0);
        target.draw(
            &self.vertex_buffer,
            &self.index_buffer,
            &self.program,
            &uniforms,
            &Default::default(),
        ).unwrap();
        target.finish().unwrap();
    }

    fn update_timer(&mut self) {
        let fixed_time_stamp = 1_000_000_000 / FPS;
        let now = time::precise_time_ns();
        self.accumulator += now - self.previous_clock;
        self.previous_clock = now;
        while self.accumulator >= fixed_time_stamp {
            self.accumulator -= fixed_time_stamp;
            // TODO: update realtime state here
        }
        let remainder_ms = (fixed_time_stamp - self.accumulator) / 1_000_000;
        thread::sleep(Duration::from_millis(remainder_ms));
    }
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let mut visualizer = Visualizer::new();
    while visualizer.is_running() {
        visualizer.draw();
        visualizer.handle_events();
        visualizer.update_timer();
    }
}

// vim: set tabstop=4 shiftwidth=4 softtabstop=4 expandtab:
