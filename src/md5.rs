// See LICENSE file for copyright and license details.

use std::fmt::{Debug};
use std::io::{BufRead};
use std::path::{Path};
use std::str::{SplitWhitespace, FromStr};
use cgmath::{Vector2, Vector3, Quaternion, Rotation};
use fs;
use ::Vertex;

#[derive(Debug, Clone)]
struct VertexWeightIndices {
    first_weight_index: usize,
    weight_count: usize,
}

#[derive(Debug, Clone)]
struct Weight {
    joint_index: usize,
    weight: f32,
    position: Vector3<f32>,
}

#[derive(Debug)]
pub struct Mesh {
    shader: String,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    max_joints_per_vert: usize,
    weights: Vec<Weight>,
    vertex_weight_indices: Vec<VertexWeightIndices>,
}

impl Mesh {
    pub fn shader(&self) -> &str {
        &self.shader
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Compute real points from bones data.
    fn calc_points(&mut self, joints: &[Joint]) {
        for i in 0..self.vertices.len() {
            let current_vertex = &self.vertex_weight_indices[i];
            let mut p = Vector3{x: 0.0, y: 0.0, z: 0.0};
            for k in 0..current_vertex.weight_count {
                let w = &self.weights[current_vertex.first_weight_index + k];
                let j = &joints[w.joint_index];
                p += j.transform(&w.position) * w.weight;
            }
            self.vertices[i].position = p.into();
        }
    }
}

#[derive(Debug, Clone)]
pub struct Joint {
    name: String,
    parent_index: Option<usize>,
    position: Vector3<f32>,
    orient: Quaternion<f32>,
}

impl Joint {
    fn transform(&self, v: &Vector3<f32>) -> Vector3<f32> {
        self.orient.rotate_vector(*v) + self.position
    }
}

#[derive(Debug)]
pub struct Model {
    joints: Vec<Joint>,
    meshes: Vec<Mesh>,
}

impl Model {
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn compute(&mut self, joints: &[Joint]) {
        for mesh in &mut self.meshes {
            mesh.calc_points(joints);
        }
    }
}

#[derive(Debug, Clone)]
struct HierarchyItem {
    name: String,
    parent: Option<usize>,
    flags: i32,
    start_index: usize,
}

#[derive(Debug, Clone)]
struct BaseFrameJoint {
    position: Vector3<f32>,
    orient: Quaternion<f32>,
}

#[derive(Debug, Clone)]
pub struct Anim {
    hierarchy: Vec<HierarchyItem>,
    base_frame: Vec<BaseFrameJoint>,
    frames: Vec<Vec<f32>>,
    joints: Vec<Joint>,
    num_animated_components: usize,
    frame: usize,
}

impl Anim {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    fn reset_joints(&mut self) {
        for i in 0..self.joints.len() {
            let j = &mut self.joints[i];
            let f = &self.base_frame[i];
            j.position = f.position;
            j.orient = f.orient;
        }
    }

    fn build_joints(&mut self) {
        for i in 0..self.joints.len() {
            if let Some(parent_index) = self.joints[i].parent_index {
                let p = self.joints[parent_index].clone();
                let j = &mut self.joints[i];
                j.position = p.position + p.orient.rotate_vector(j.position);
                j.orient = p.orient * j.orient;
            }
        }
    }

    pub fn set_frame(&mut self, n: usize) {
        assert!(n < self.frames.len());
        self.frame = n;
        self.reset_joints();
        for i in 0..self.joints.len() {
            let flags = self.hierarchy[i].flags;
            let mut position = self.hierarchy[i].start_index;
            let j = &mut self.joints[i];
            if flags & 1 != 0 {
                j.position.x = self.frames[n][position];
                position += 1;
            }
            if flags & 2 != 0 {
                j.position.y = self.frames[n][position];
                position += 1;
            }
            if flags & 4 != 0 {
                j.position.z = self.frames[n][position];
                position += 1;
            }
            if flags & 8 != 0 {
                j.orient.v.x = self.frames[n][position];
                position += 1;
            }
            if flags & 16 != 0 {
                j.orient.v.y = self.frames[n][position];
                position += 1;
            }
            if flags & 32 != 0 {
                j.orient.v.z = self.frames[n][position];
            }
            j.orient = compute_quat_w(j.orient.v);
        }
        self.build_joints();
    }
}

fn parse_word<T: FromStr>(words: &mut SplitWhitespace) -> T
    where T::Err: Debug
{
    let str = words.next().expect("Can not read next word");
    str.parse().expect("Can not parse word")
}

fn expect_word(words: &mut SplitWhitespace, expected: &str) {
    let str = words.next().expect("Can not read next word");
    if str != expected {
        panic!("Expected '{}', got '{}'", expected, str);
    }
}

fn read_mesh(buf: &mut BufRead) -> Mesh {
    let mut m = Mesh {
        indices: Vec::new(),
        vertices: Vec::new(),
        vertex_weight_indices: Vec::new(),
        weights: Vec::new(),
        shader: "".into(),
        max_joints_per_vert: 0,
    };
    for line in buf.lines() {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        if let Some(tag) = words.next() {
            if tag == "}" {
                break;
            }
            if tag == "numverts" {
                let num_vertices = parse_word(&mut words);
                m.vertices.reserve(num_vertices);
            }
            if tag == "numtris" {
                let num_tris: usize = parse_word(&mut words);
                m.indices.reserve(num_tris * 3)
            }
            if tag == "numweights" {
                let num_weights = parse_word(&mut words);
                m.weights.reserve(num_weights)
            }
            if tag == "vert" {
                let index: usize = parse_word(&mut words);
                expect_word(&mut words, "(");
                let tex_coords = Vector2 {
                    x: parse_word(&mut words),
                    y: parse_word(&mut words),
                };
                expect_word(&mut words, ")");
                m.vertices.push(Vertex {
                    tex_coords: tex_coords.into(),
                    position: [0.0, 0.0, 0.0],
                });
                assert_eq!(m.vertices.len() - 1, index);
                m.vertex_weight_indices.push(VertexWeightIndices {
                    first_weight_index: parse_word(&mut words),
                    weight_count: parse_word(&mut words),
                });
                assert_eq!(m.vertex_weight_indices.len() - 1, index);
            }
            if tag == "weight" {
                let index: usize = parse_word(&mut words);
                let joint_index = parse_word(&mut words);
                let weight = parse_word(&mut words);
                expect_word(&mut words, "(");
                m.weights.push(Weight {
                    joint_index: joint_index,
                    weight: weight,
                    position: Vector3 {
                        x: parse_word(&mut words),
                        y: parse_word(&mut words),
                        z: parse_word(&mut words),
                    },
                });
                expect_word(&mut words, ")");
                assert_eq!(m.weights.len() - 1, index);
            }
            if tag == "tri" {
                let index: usize = parse_word(&mut words);
                m.indices.push(parse_word(&mut words));
                m.indices.push(parse_word(&mut words));
                m.indices.push(parse_word(&mut words));
                assert_eq!(m.indices.len() - 3, index * 3);
            }
            if tag == "shader" {
                m.shader = words.next().unwrap().trim_matches('"').replace("data/", "");
                m.shader = format!("{}.png", m.shader);
            }
        }
    }
    for wi in &m.vertex_weight_indices {
        if wi.weight_count > m.max_joints_per_vert {
            m.max_joints_per_vert = wi.weight_count;
        }
    }
    m
}

fn compute_quat_w(v: Vector3<f32>) -> Quaternion<f32> {
    let t = 1.0 - (v.x * v.x) - (v.y * v.y) - (v.z * v.z);
    let w = if t < 0.0 {
        0.0
    } else {
        -t.sqrt()
    };
    Quaternion::from_sv(w, v)
}

fn read_joints(buf: &mut BufRead) -> Vec<Joint> {
    let mut joints = Vec::new();
    for line in buf.lines() {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        if let Some(tag) = words.next() {
            if tag == "}" {
                break;
            }
            let name = tag.trim_matches('"').into();
            let parent_index: isize = parse_word(&mut words);
            expect_word(&mut words, "(");
            let position = Vector3 {
                x: parse_word(&mut words),
                y: parse_word(&mut words),
                z: parse_word(&mut words),
            };
            expect_word(&mut words, ")");
            expect_word(&mut words, "(");
            let q = compute_quat_w(Vector3 {
                x: parse_word(&mut words),
                y: parse_word(&mut words),
                z: parse_word(&mut words),
            });
            expect_word(&mut words, ")");
            joints.push(Joint {
                name: name,
                parent_index: if parent_index != -1 {
                    Some(parent_index as usize)
                } else {
                    None
                },
                position: position,
                orient: q,
            });
        }
    }
    joints
}

fn read_line(buf: &mut BufRead) -> Option<String> {
    let mut line = "".into();
    if let Ok(bytes) = buf.read_line(&mut line) {
        if bytes == 0 {
            None
        } else {
            Some(line)
        }
    } else {
        None
    }
}

pub fn load_model<P: AsRef<Path>>(path: P) -> Model {
    let mut model = Model {
        joints: Vec::new(),
        meshes: Vec::new(),
    };
    let mut buf = fs::load(path);
    while let Some(line) = read_line(&mut buf) {
        let mut words = line.split_whitespace();
        if let Some(tag) = words.next() {
            if tag == "numJoints" {
                let num_joints = parse_word(&mut words);
                model.joints.reserve(num_joints);
            }
            if tag == "numMeshes" {
                let num_meshes = parse_word(&mut words);
                model.meshes.reserve(num_meshes);
            }
            if tag == "joints" {
                expect_word(&mut words, "{");
                model.joints = read_joints(&mut buf);
            }
            if tag == "mesh" {
                expect_word(&mut words, "{");
                let mesh = read_mesh(&mut buf);
                model.meshes.push(mesh);
            }
            // } else {
            //     // puts("...");
            // }
        }
    }
    for mesh in &mut model.meshes {
        mesh.calc_points(&model.joints); // T-pose
    }
    model
}

fn load_hierarchy(buf: &mut BufRead) -> Vec<HierarchyItem> {
    let mut hierarchy = Vec::new();
    for line in buf.lines() {
        let line = line.unwrap();
        if line.trim() == "}" {
            break;
        }
        let mut words = line.split_whitespace();
        let name: String = words.next().unwrap().trim_matches('"').into();
        let parent: isize = parse_word(&mut words);
        let flags: i32 = parse_word(&mut words);
        let start_index: usize = parse_word(&mut words);
        hierarchy.push(HierarchyItem {
            name: name,
            parent: if parent != -1 {
                Some(parent as usize)
            } else {
                None
            },
            flags: flags,
            start_index: start_index,
        });
    }
    hierarchy
}

fn load_base_frame(buf: &mut BufRead) -> Vec<BaseFrameJoint> {
    let mut frame = Vec::new();
    for line in buf.lines() {
        let line = line.unwrap();
        if line.trim() == "}" {
            break;
        }
        let mut words = line.split_whitespace();
        expect_word(&mut words, "(");
        let pos: Vector3<f32> = Vector3 {
            x: parse_word(&mut words),
            y: parse_word(&mut words),
            z: parse_word(&mut words),
        };
        expect_word(&mut words, ")");
        expect_word(&mut words, "(");
        let q = compute_quat_w(Vector3 {
            x: parse_word(&mut words),
            y: parse_word(&mut words),
            z: parse_word(&mut words),
        });
        expect_word(&mut words, ")");
        frame.push(BaseFrameJoint{position: pos, orient: q});
    }
    frame
}

fn load_frame(buf: &mut BufRead, num_animated_components: usize) -> Vec<f32> {
    let mut frame = Vec::with_capacity(num_animated_components);
    for line in buf.lines() {
        let line = line.unwrap();
        if line.trim() == "}" {
            break;
        }
        let mut words = line.split_whitespace();
        frame.push(parse_word(&mut words));
        frame.push(parse_word(&mut words));
        frame.push(parse_word(&mut words));
        frame.push(parse_word(&mut words));
        frame.push(parse_word(&mut words));
        frame.push(parse_word(&mut words));
    }
    frame
}

pub fn load_anim<P: AsRef<Path>>(path: P) -> Anim {
    let mut buf = fs::load(path);
    let mut anim = Anim {
        hierarchy: Vec::new(),
        base_frame: Vec::new(),
        frames: Vec::new(),
        joints: Vec::new(),
        num_animated_components: 0,
        frame: 0,
    };
    let mut num_joints: usize = 0;
    while let Some(line) = read_line(&mut buf) {
        let mut words = line.split_whitespace();
        if let Some(tag) = words.next() {
            if tag == "numFrames" {
                let num_frames: usize = parse_word(&mut words);
                anim.frames.reserve(num_frames);
            }
            if tag == "numJoints" {
                num_joints = parse_word(&mut words);
                anim.hierarchy.reserve(num_joints);
                anim.base_frame.reserve(num_joints);
                anim.joints.reserve(num_joints);
            }
            if tag == "frameRate" {
                let _: usize = parse_word(&mut words); // TODO: interpolation!
            }
            if tag == "numAnimatedComponents" {
                anim.num_animated_components = parse_word(&mut words);
            }
            if tag == "hierarchy" {
                expect_word(&mut words, "{");
                anim.hierarchy = load_hierarchy(&mut buf);
            }
            if tag == "bounds" {
                expect_word(&mut words, "{");
                // not implemented
            }
            if tag == "baseframe" {
                expect_word(&mut words, "{");
                anim.base_frame = load_base_frame(&mut buf);
            }
            if tag == "frame" {
                let index: usize = parse_word(&mut words);
                expect_word(&mut words, "{");
                anim.frames.push(load_frame(
                    &mut buf, anim.num_animated_components));
                assert_eq!(anim.frames.len() - 1, index);
            }
        }
    }
    anim.joints.reserve(num_joints);
    for i in 0..num_joints {
        anim.joints.push(Joint {
            name: anim.hierarchy[i].name.clone(),
            parent_index: anim.hierarchy[i].parent,
            position: anim.base_frame[i].position,
            orient: anim.base_frame[i].orient,
        });
    }
    anim.reset_joints();
    anim.build_joints();
    anim
}

// vim: set tabstop=4 shiftwidth=4 softtabstop=4 expandtab:
