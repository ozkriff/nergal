// See LICENSE file for copyright and license details.

use std::fmt::{Debug};
use std::io::{BufRead};
use std::path::{Path, PathBuf};
use std::str::{SplitWhitespace, FromStr};
use cgmath::{Vector3, Quaternion, Rotation, InnerSpace};
use fs;

const BIT_POS_X: i32 = 1;
const BIT_POS_Y: i32 = 2;
const BIT_POS_Z: i32 = 4;
const BIT_QUAT_X: i32 = 8;
const BIT_QUAT_Y: i32 = 16;
const BIT_QUAT_Z: i32 = 32;

#[derive(Debug, Copy, Clone)]
pub struct VertexPos {
    pub position: [f32; 3],
}

// Separate from VertexPos because we don't need to update UV every frame
#[derive(Debug, Copy, Clone)]
pub struct VertexUV {
    pub uv: [f32; 2],
}

#[derive(Debug, Copy, Clone)]
struct VertexWeightIndices {
    first_weight_index: usize,
    weight_count: usize,
}

#[derive(Debug, Copy, Clone)]
struct Weight {
    joint_index: usize,
    weight: f32,
    position: Vector3<f32>,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    texture_path: PathBuf,
    vertex_positions: Vec<VertexPos>,
    vertex_uvs: Vec<VertexUV>,
    indices: Vec<u16>,
    max_joints_per_vert: usize,
    weights: Vec<Weight>,
    vertex_weight_indices: Vec<VertexWeightIndices>,
}

impl Mesh {
    fn new(buf: &mut BufRead) -> Mesh {
        let mut m = Mesh {
            indices: Vec::new(),
            vertex_positions: Vec::new(),
            vertex_uvs: Vec::new(),
            vertex_weight_indices: Vec::new(),
            weights: Vec::new(),
            texture_path: PathBuf::new(),
            max_joints_per_vert: 0,
        };
        for line in buf.lines() {
            let line = line.unwrap();
            let mut words = line.split_whitespace();
            let tag = match words.next() {
                Some(tag) => tag,
                None => continue,
            };
            match tag {
                "}" => {
                    break;
                }
                "numverts" => {
                    let num_vertices = parse_word(&mut words);
                    m.vertex_positions.reserve(num_vertices);
                }
                "numtris" => {
                    let num_tris: usize = parse_word(&mut words);
                    m.indices.reserve(num_tris * 3)
                }
                "numweights" => {
                    let num_weights = parse_word(&mut words);
                    m.weights.reserve(num_weights)
                }
                "vert" => {
                    let index = parse_word(&mut words);
                    expect_word(&mut words, "(");
                    let uv = [
                        parse_word(&mut words),
                        parse_word(&mut words),
                    ];
                    expect_word(&mut words, ")");
                    m.vertex_uvs.push(VertexUV{uv: uv.into()});
                    m.vertex_positions.push(VertexPos{position: [0.0; 3]});
                    assert_eq!(m.vertex_positions.len() - 1, index);
                    m.vertex_weight_indices.push(VertexWeightIndices {
                        first_weight_index: parse_word(&mut words),
                        weight_count: parse_word(&mut words),
                    });
                    assert_eq!(m.vertex_weight_indices.len() - 1, index);
                }
                "weight" => {
                    let index = parse_word(&mut words);
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
                "tri" => {
                    let index: usize = parse_word(&mut words);
                    m.indices.push(parse_word(&mut words));
                    m.indices.push(parse_word(&mut words));
                    m.indices.push(parse_word(&mut words));
                    assert_eq!(m.indices.len() - 3, index * 3);
                }
                "shader" => {
                    let texture_name = words.next().unwrap().trim_matches('"').replace("data/", "");
                    m.texture_path = PathBuf::from(format!("{}.png", texture_name));
                }
                unexpected_tag => {
                    println!("load_mesh: unexpected tag: {}", unexpected_tag);
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

    pub fn texture_path(&self) -> &Path {
        &self.texture_path
    }

    pub fn vertex_positions(&self) -> &[VertexPos] {
        &self.vertex_positions
    }

    pub fn vertex_uvs(&self) -> &[VertexUV] {
        &self.vertex_uvs
    }

    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    fn update_vertex_positions(&mut self, joints: &[Joint]) {
        for i in 0..self.vertex_positions.len() {
            let current_vertex = &self.vertex_weight_indices[i];
            let mut p = Vector3{x: 0.0, y: 0.0, z: 0.0};
            for k in 0..current_vertex.weight_count {
                let w = &self.weights[current_vertex.first_weight_index + k];
                let j = &joints[w.joint_index];
                p += j.transform(&w.position) * w.weight;
            }
            self.vertex_positions[i].position = p.into();
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

#[derive(Debug, Clone)]
pub struct Model {
    joints: Vec<Joint>,
    meshes: Vec<Mesh>,
}

impl Model {
    pub fn new<P: AsRef<Path>>(path: P) -> Model {
        let mut model = Model {
            joints: Vec::new(),
            meshes: Vec::new(),
        };
        let mut buf = fs::load(path);
        while let Some(line) = read_line(&mut buf) {
            let mut words = line.split_whitespace();
            let tag = match words.next() {
                Some(tag) => tag,
                None => continue,
            };
            match tag {
                "numJoints" => {
                    let num_joints = parse_word(&mut words);
                    model.joints.reserve(num_joints);
                }
                "numMeshes" => {
                    let num_meshes = parse_word(&mut words);
                    model.meshes.reserve(num_meshes);
                }
                "joints" => {
                    expect_word(&mut words, "{");
                    model.joints = Anim::load_joints(&mut buf);
                }
                "mesh" => {
                    expect_word(&mut words, "{");
                    let mesh = Mesh::new(&mut buf);
                    model.meshes.push(mesh);
                }
                "MD5Version" => {
                    // unused
                }
                "commandline" => {
                    // unused
                }
                unexpected_tag => {
                    println!("load_model: unexpected tag: {}", unexpected_tag);
                }
            }
        }
        for mesh in &mut model.meshes {
            mesh.update_vertex_positions(&model.joints); // T-pose
        }
        model
    }

    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn update_vertex_positions(&mut self, joints: &[Joint]) {
        for mesh in &mut self.meshes {
            mesh.update_vertex_positions(joints);
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AnimationMode {
    KeyframesOnly,
    Interpolation,
}

#[derive(Debug, Clone)]
pub struct Anim {
    hierarchy: Vec<HierarchyItem>,
    base_frame: Vec<BaseFrameJoint>,
    frames: Vec<Vec<f32>>,
    joints: Vec<Joint>,
    num_animated_components: usize,
    frame_rate: usize,
    time: f32,
    joints_prev: Vec<Joint>,
    joints_next: Vec<Joint>,
    mode: AnimationMode,
}

impl Anim {
    pub fn new<P: AsRef<Path>>(path: P, mode: AnimationMode) -> Anim {
        let mut buf = fs::load(path);
        let mut anim = Anim {
            hierarchy: Vec::new(),
            base_frame: Vec::new(),
            frames: Vec::new(),
            joints: Vec::new(),
            joints_prev: Vec::new(),
            joints_next: Vec::new(),
            num_animated_components: 0,
            frame_rate: 0,
            time: 0.0,
            mode: mode,
        };
        let mut num_joints = 0;
        while let Some(line) = read_line(&mut buf) {
            let mut words = line.split_whitespace();
            let tag = match words.next() {
                Some(tag) => tag,
                None => continue,
            };
            match tag {
                "numFrames" => {
                    let num_frames = parse_word(&mut words);
                    anim.frames.reserve(num_frames);
                }
                "numJoints" => {
                    num_joints = parse_word(&mut words);
                    anim.hierarchy.reserve(num_joints);
                    anim.base_frame.reserve(num_joints);
                    anim.joints.reserve(num_joints);
                }
                "frameRate" => {
                    anim.frame_rate = parse_word(&mut words);
                    anim.frame_rate /= 15; // TODO: for debugging
                }
                "numAnimatedComponents" => {
                    anim.num_animated_components = parse_word(&mut words);
                }
                "hierarchy" => {
                    expect_word(&mut words, "{");
                    anim.hierarchy = Anim::load_hierarchy(&mut buf);
                }
                "bounds" => {
                    expect_word(&mut words, "{");
                    let _ = Anim::load_bounds(&mut buf);
                }
                "baseframe" => {
                    expect_word(&mut words, "{");
                    anim.base_frame = Anim::load_base_frame(&mut buf);
                }
                "frame" => {
                    let index = parse_word(&mut words);
                    expect_word(&mut words, "{");
                    anim.frames.push(Anim::load_frame(
                        &mut buf, anim.num_animated_components));
                    assert_eq!(anim.frames.len() - 1, index);
                }
                "MD5Version" => {
                    // unused
                }
                "commandline" => {
                    // unused
                }
                unexpected_tag => {
                    println!("load_anim: unexpected tag: {}", unexpected_tag);
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
        anim.joints_prev = anim.joints.clone();
        if mode == AnimationMode::Interpolation {
            anim.joints_next = anim.joints.clone();
        }
        Anim::reset_joints(&anim.base_frame, &mut anim.joints);
        Anim::build_joints(&mut anim.joints);
        anim
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

    pub fn len(&self) -> f32 {
        self.frames.len() as f32 / self.frame_rate as f32
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    fn wrap_frame_index(&self, n: usize) -> usize {
        let mut n = n;
        while n >= self.frames.len() {
            n -= self.frames.len();
        }
        n
    }

    fn reset_joints(base_frame: &[BaseFrameJoint], joints: &mut [Joint]) {
        for i in 0..joints.len() {
            let j = &mut joints[i];
            let f = &base_frame[i];
            j.position = f.position;
            j.orient = f.orient;
        }
    }

    fn build_joints(joints: &mut [Joint]) {
        for i in 0..joints.len() {
            if let Some(parent_index) = joints[i].parent_index {
                let p = joints[parent_index].clone();
                let j = &mut joints[i];
                j.position = p.position + p.orient.rotate_vector(j.position);
                j.orient = p.orient * j.orient;
            }
        }
    }

    pub fn set_time(&mut self, time: f32) {
        self.time = time;
        self.update(0.0);
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        let t = self.time * self.frame_rate as f32;
        let factor = t % 1.0;
        let prev_frame_index = self.wrap_frame_index(t as usize);
        Anim::update_internal(
            &self.base_frame,
            &self.hierarchy,
            &self.frames,
            prev_frame_index,
            &mut self.joints_prev,
        );
        if self.mode == AnimationMode::Interpolation {
            let next_frame_index = self.wrap_frame_index(prev_frame_index + 1);
            Anim::update_internal(
                &self.base_frame,
                &self.hierarchy,
                &self.frames,
                next_frame_index,
                &mut self.joints_next,
            );
            for i in 0..self.joints.len() {
                let j_prev = &self.joints_prev[i];
                let j_next = &self.joints_next[i];
                let j = &mut self.joints[i];
                j.position = j_prev.position.lerp(j_next.position, factor);
                j.orient = j_prev.orient.slerp(j_next.orient, factor); // TODO: nlerp?
            }
        } else {
            self.joints = self.joints_prev.clone();
        }
    }

    fn update_internal(
        base_frame: &[BaseFrameJoint],
        hierarchy: &[HierarchyItem],
        frames: &[Vec<f32>],
        n: usize,
        joints: &mut [Joint],
    ) {
        Anim::reset_joints(base_frame, joints);
        let f = &frames[n];
        for i in 0..joints.len() {
            let flags = hierarchy[i].flags;
            let mut index_cursor = hierarchy[i].start_index;
            let j = &mut joints[i];
            if flags & BIT_POS_X != 0 {
                j.position.x = f[index_cursor];
                index_cursor += 1;
            }
            if flags & BIT_POS_Y != 0 {
                j.position.y = f[index_cursor];
                index_cursor += 1;
            }
            if flags & BIT_POS_Z != 0 {
                j.position.z = f[index_cursor];
                index_cursor += 1;
            }
            if flags & BIT_QUAT_X != 0 {
                j.orient.v.x = f[index_cursor];
                index_cursor += 1;
            }
            if flags & BIT_QUAT_Y != 0 {
                j.orient.v.y = f[index_cursor];
                index_cursor += 1;
            }
            if flags & BIT_QUAT_Z != 0 {
                j.orient.v.z = f[index_cursor];
            }
            j.orient = compute_quat_w(j.orient.v);
        }
        Anim::build_joints(joints);
    }

    fn load_joints(buf: &mut BufRead) -> Vec<Joint> {
        let mut joints = Vec::new();
        for line in buf.lines() {
            let line = line.unwrap();
            let mut words = line.split_whitespace();
            let tag = match words.next() {
                Some(tag) => tag,
                None => continue,
            };
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
        joints
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
            let start_index = parse_word(&mut words);
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

    fn load_bounds(buf: &mut BufRead) {
        for line in buf.lines() {
            let line = line.unwrap();
            if line.trim() == "}" {
                break;
            }
            // unused
        }
    }
}

fn read_line(buf: &mut BufRead) -> Option<String> {
    let mut line = "".into();
    match buf.read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
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

fn compute_quat_w(v: Vector3<f32>) -> Quaternion<f32> {
    let t = 1.0 - (v.x * v.x) - (v.y * v.y) - (v.z * v.z);
    let w = if t < 0.0 {
        0.0
    } else {
        -t.sqrt()
    };
    Quaternion::from_sv(w, v)
}

// vim: set tabstop=4 shiftwidth=4 softtabstop=4 expandtab:
