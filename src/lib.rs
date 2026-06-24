use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;

const SIZE_INT: usize = 4;
const SIZE_FLOAT: usize = 4;
const LINE_LEN: usize = 80;

#[derive(Debug)]
pub struct EnsightReaderError {
    pub message: String,
    pub file_path: Option<String>,
    pub file_offset: Option<u64>,
    pub file_lineno: Option<usize>,
}

impl EnsightReaderError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            file_path: None,
            file_offset: None,
            file_lineno: None,
        }
    }

    fn with_context(
        message: impl Into<String>,
        path: Option<&Path>,
        offset: Option<u64>,
        lineno: Option<usize>,
    ) -> Self {
        Self {
            message: message.into(),
            file_path: path.map(|p: &Path| p.to_string_lossy().to_string()),
            file_offset: offset,
            file_lineno: lineno,
        }
    }
}

impl std::fmt::Display for EnsightReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.file_lineno {
            write!(f, "{} (line={})", self.message, line)
        } else if let Some(offset) = self.file_offset {
            write!(f, "{} (offset={})", self.message, offset)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for EnsightReaderError {}

pub type Result<T> = std::result::Result<T, EnsightReaderError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdHandling {
    Off,
    Given,
    Assign,
    Ignore,
}

impl IdHandling {
    fn ids_present(self) -> bool {
        matches!(self, IdHandling::Given | IdHandling::Ignore)
    }
}

impl std::str::FromStr for IdHandling {
    type Err = EnsightReaderError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(IdHandling::Off),
            "given" => Ok(IdHandling::Given),
            "assign" => Ok(IdHandling::Assign),
            "ignore" => Ok(IdHandling::Ignore),
            _ => Err(EnsightReaderError::new(format!("Unknown id handling: {s}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableLocation {
    PerElement,
    PerNode,
}

impl std::str::FromStr for VariableLocation {
    type Err = EnsightReaderError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "element" => Ok(VariableLocation::PerElement),
            "node" => Ok(VariableLocation::PerNode),
            _ => Err(EnsightReaderError::new(format!(
                "Unknown variable location: {s}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Scalar,
    Vector,
    TensorSymm,
    TensorAsym,
}

impl VariableType {
    fn value_count(self) -> usize {
        match self {
            VariableType::Scalar => 1,
            VariableType::Vector => 3,
            VariableType::TensorSymm => 6,
            VariableType::TensorAsym => 9,
        }
    }
}

impl std::str::FromStr for VariableType {
    type Err = EnsightReaderError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "scalar" => Ok(VariableType::Scalar),
            "vector" => Ok(VariableType::Vector),
            "tensor symm" => Ok(VariableType::TensorSymm),
            "tensor asym" => Ok(VariableType::TensorAsym),
            _ => Err(EnsightReaderError::new(format!(
                "Unknown variable type: {s}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    Point,
    Bar2,
    Bar3,
    Tria3,
    Tria6,
    Quad4,
    Quad8,
    Tetra4,
    Tetra10,
    Pyramid5,
    Pyramid13,
    Penta6,
    Penta15,
    Hexa8,
    Hexa20,
    Nsided,
    Nfaced,
}

impl ElementType {
    fn parse_from_line(line: &str) -> Result<Self> {
        let token: &str = line
            .split_whitespace()
            .next()
            .ok_or_else(|| EnsightReaderError::new("Empty element type line"))?;
        match token {
            "point" => Ok(ElementType::Point),
            "bar2" => Ok(ElementType::Bar2),
            "bar3" => Ok(ElementType::Bar3),
            "tria3" => Ok(ElementType::Tria3),
            "tria6" => Ok(ElementType::Tria6),
            "quad4" => Ok(ElementType::Quad4),
            "quad8" => Ok(ElementType::Quad8),
            "tetra4" => Ok(ElementType::Tetra4),
            "tetra10" => Ok(ElementType::Tetra10),
            "pyramid5" => Ok(ElementType::Pyramid5),
            "pyramid13" => Ok(ElementType::Pyramid13),
            "penta6" => Ok(ElementType::Penta6),
            "penta15" => Ok(ElementType::Penta15),
            "hexa8" => Ok(ElementType::Hexa8),
            "hexa20" => Ok(ElementType::Hexa20),
            "nsided" => Ok(ElementType::Nsided),
            "nfaced" => Ok(ElementType::Nfaced),
            _ => Err(EnsightReaderError::new(format!(
                "Unsupported element type: {token}"
            ))),
        }
    }

    pub fn nodes_per_element(self) -> Option<usize> {
        match self {
            ElementType::Point => Some(1),
            ElementType::Bar2 => Some(2),
            ElementType::Bar3 => Some(3),
            ElementType::Tria3 => Some(3),
            ElementType::Tria6 => Some(6),
            ElementType::Quad4 => Some(4),
            ElementType::Quad8 => Some(8),
            ElementType::Tetra4 => Some(4),
            ElementType::Tetra10 => Some(10),
            ElementType::Pyramid5 => Some(5),
            ElementType::Pyramid13 => Some(13),
            ElementType::Penta6 => Some(6),
            ElementType::Penta15 => Some(15),
            ElementType::Hexa8 => Some(8),
            ElementType::Hexa20 => Some(20),
            ElementType::Nsided | ElementType::Nfaced => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Timeset {
    pub timeset_id: i32,
    pub description: Option<String>,
    pub number_of_steps: usize,
    pub filename_numbers: Vec<i32>,
    pub time_values: Vec<f64>,
}

impl Timeset {
    fn filename_numbers_from_arithmetic_sequence(
        file_start_number: i32,
        number_of_steps: usize,
        filename_increment: i32,
    ) -> Vec<i32> {
        (0..number_of_steps)
            .map(|i: usize| file_start_number + (i as i32) * filename_increment)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct GeometryPart {
    pub offset: u64,
    pub part_id: i32,
    pub part_name: String,
    pub number_of_nodes: usize,
    pub element_blocks: Vec<UnstructuredElementBlock>,
    node_id_handling: IdHandling,
    element_id_handling: IdHandling,
}

impl GeometryPart {
    pub fn read_nodes(&self, fp: &mut Reader) -> Result<Vec<f32>> {
        fp.seek(SeekFrom::Start(self.offset))
            .map_err(|e| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
        let part_line: String = read_line(fp)?;
        if !part_line.starts_with("part") {
            return Err(EnsightReaderError::new("Expected 'part' line"));
        }
        let part_id: i32 = read_i32(fp)?;
        if part_id != self.part_id {
            return Err(EnsightReaderError::new("Part id mismatch"));
        }
        let _ = read_line(fp)?; // part name line
        let coordinates_line: String = read_line(fp)?;
        if !coordinates_line.starts_with("coordinates") {
            return Err(EnsightReaderError::new("Expected 'coordinates' line"));
        }
        let number_of_nodes: usize = read_i32(fp)? as usize;
        if number_of_nodes != self.number_of_nodes {
            return Err(EnsightReaderError::new("Node count mismatch"));
        }
        if self.node_id_handling.ids_present() {
            skip_bytes(fp, number_of_nodes * SIZE_INT)?;
        }
        read_f32_vec(fp, 3 * number_of_nodes)
    }
}

#[derive(Debug, Clone)]
pub struct UnstructuredElementBlock {
    pub offset: u64,
    pub number_of_elements: usize,
    pub element_type: ElementType,
    element_id_handling: IdHandling,
    pub part_id: i32,
}

impl UnstructuredElementBlock {
    pub fn read_connectivity(&self, fp: &mut Reader) -> Result<Vec<i32>> {
        let nodes_per_element: usize = self.element_type.nodes_per_element().ok_or_else(|| {
            EnsightReaderError::new("Use nsided/nfaced methods for variable-sized elements")
        })?;
        fp.seek(SeekFrom::Start(self.offset))
            .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
        let element_type_line: String = read_line(fp)?;
        let parsed: ElementType = ElementType::parse_from_line(&element_type_line)?;
        if parsed != self.element_type {
            return Err(EnsightReaderError::new("Element type mismatch"));
        }
        let number_of_elements: usize = read_i32(fp)? as usize;
        if number_of_elements != self.number_of_elements {
            return Err(EnsightReaderError::new("Element count mismatch"));
        }
        if self.element_id_handling.ids_present() {
            skip_bytes(fp, number_of_elements * SIZE_INT)?;
        }
        read_i32_vec(fp, number_of_elements * nodes_per_element)
    }

    pub fn read_connectivity_nsided(&self, fp: &mut Reader) -> Result<(Vec<i32>, Vec<i32>)> {
        if self.element_type != ElementType::Nsided {
            return Err(EnsightReaderError::new("Element type is not nsided"));
        }
        fp.seek(SeekFrom::Start(self.offset))
            .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
        let element_type_line: String = read_line(fp)?;
        let parsed: ElementType = ElementType::parse_from_line(&element_type_line)?;
        if parsed != self.element_type {
            return Err(EnsightReaderError::new("Element type mismatch"));
        }
        let number_of_elements: usize = read_i32(fp)? as usize;
        if number_of_elements != self.number_of_elements {
            return Err(EnsightReaderError::new("Element count mismatch"));
        }
        if self.element_id_handling.ids_present() {
            skip_bytes(fp, number_of_elements * SIZE_INT)?;
        }
        let polygon_node_counts: Vec<i32> = read_i32_vec(fp, number_of_elements)?;
        let total_nodes: usize = polygon_node_counts.iter().map(|v: &i32| *v as usize).sum();
        let polygon_connectivity: Vec<i32> = read_i32_vec(fp, total_nodes)?;
        Ok((polygon_node_counts, polygon_connectivity))
    }

    pub fn read_connectivity_nfaced(
        &self,
        fp: &mut Reader,
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        if self.element_type != ElementType::Nfaced {
            return Err(EnsightReaderError::new("Element type is not nfaced"));
        }
        fp.seek(SeekFrom::Start(self.offset))
            .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
        let element_type_line: String = read_line(fp)?;
        let parsed: ElementType = ElementType::parse_from_line(&element_type_line)?;
        if parsed != self.element_type {
            return Err(EnsightReaderError::new("Element type mismatch"));
        }
        let number_of_elements: usize = read_i32(fp)? as usize;
        if number_of_elements != self.number_of_elements {
            return Err(EnsightReaderError::new("Element count mismatch"));
        }
        if self.element_id_handling.ids_present() {
            skip_bytes(fp, number_of_elements * SIZE_INT)?;
        }
        let polyhedra_face_counts: Vec<i32> = read_i32_vec(fp, number_of_elements)?;
        let total_faces: usize = polyhedra_face_counts
            .iter()
            .map(|v: &i32| *v as usize)
            .sum();
        let face_node_counts: Vec<i32> = read_i32_vec(fp, total_faces)?;
        let total_nodes: usize = face_node_counts.iter().map(|v: &i32| *v as usize).sum();
        let face_connectivity: Vec<i32> = read_i32_vec(fp, total_nodes)?;
        Ok((polyhedra_face_counts, face_node_counts, face_connectivity))
    }

    fn from_file(fp: &mut Reader, element_id_handling: IdHandling, part_id: i32) -> Result<Self> {
        let offset: u64 = fp.stream_position().unwrap_or(0);
        let element_type_line: String = read_line(fp)?;
        let element_type: ElementType = ElementType::parse_from_line(&element_type_line)?;
        let number_of_elements: usize = read_i32(fp)? as usize;
        if element_id_handling.ids_present() {
            skip_bytes(fp, number_of_elements * SIZE_INT)?;
        }
        if let Some(nodes_per_element) = element_type.nodes_per_element() {
            skip_bytes(fp, number_of_elements * nodes_per_element * SIZE_INT)?;
        } else if element_type == ElementType::Nsided {
            let polygon_node_counts: Vec<i32> = read_i32_vec(fp, number_of_elements)?;
            let total_nodes: usize = polygon_node_counts.iter().map(|v: &i32| *v as usize).sum();
            skip_bytes(fp, total_nodes * SIZE_INT)?;
        } else if element_type == ElementType::Nfaced {
            let polyhedra_face_counts: Vec<i32> = read_i32_vec(fp, number_of_elements)?;
            let total_faces: usize = polyhedra_face_counts
                .iter()
                .map(|v: &i32| *v as usize)
                .sum();
            let face_node_counts: Vec<i32> = read_i32_vec(fp, total_faces)?;
            let total_nodes: usize = face_node_counts.iter().map(|v: &i32| *v as usize).sum();
            skip_bytes(fp, total_nodes * SIZE_INT)?;
        } else {
            return Err(EnsightReaderError::new("Unsupported element type"));
        }

        Ok(Self {
            offset,
            number_of_elements,
            element_type,
            element_id_handling,
            part_id,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EnsightGeometryFile {
    pub file_path: PathBuf,
    pub node_id_handling: IdHandling,
    pub element_id_handling: IdHandling,
    pub parts: Vec<GeometryPart>,
}

impl EnsightGeometryFile {
    pub fn from_file_path(
        path: impl AsRef<Path>,
        changing_geometry_per_part: bool,
    ) -> Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let mut fp: BufReader<File> = open_reader(&path)?;

        let first_line: String = read_line(&mut fp)?;
        if !first_line.to_lowercase().starts_with("c binary") {
            return Err(EnsightReaderError::new(
                "Only 'C binary' geometry files are supported",
            ));
        }
        let _ = read_line(&mut fp)?; // description 1
        let _ = read_line(&mut fp)?; // description 2

        let node_id_line: String = read_line(&mut fp)?;
        let node_id_handling: IdHandling = parse_id_handling(&node_id_line, "node id")?;

        let element_id_line: String = read_line(&mut fp)?;
        let element_id_handling: IdHandling = parse_id_handling(&element_id_line, "element id")?;

        let file_len: u64 = file_len(&mut fp)?;
        let mut parts: Vec<GeometryPart> = Vec::new();

        while fp.stream_position().unwrap_or(0) < file_len {
            let line: String = peek_line(&mut fp)?;
            if line.is_empty() {
                break;
            }
            if line.starts_with("extents") {
                let _ = read_line(&mut fp)?;
                skip_bytes(&mut fp, 6 * SIZE_FLOAT)?;
                continue;
            }
            if !line.starts_with("part") {
                return Err(EnsightReaderError::new(format!(
                    "Expected 'part' line, got: {line}"
                )));
            }
            let part: GeometryPart = GeometryPartReader::read_part(
                &mut fp,
                node_id_handling,
                element_id_handling,
                changing_geometry_per_part,
            )?;
            parts.push(part);
        }

        Ok(Self {
            file_path: path,
            node_id_handling,
            element_id_handling,
            parts,
        })
    }

    pub fn iter_parts(&self) -> impl Iterator<Item = &GeometryPart> {
        self.parts.iter()
    }

    pub fn get_part_by_id(&self, part_id: i32) -> Option<&GeometryPart> {
        self.parts
            .iter()
            .find(|p: &&GeometryPart| p.part_id == part_id)
    }
}

struct GeometryPartReader;

impl GeometryPartReader {
    fn read_part(
        fp: &mut Reader,
        node_id_handling: IdHandling,
        element_id_handling: IdHandling,
        changing_geometry_per_part: bool,
    ) -> Result<GeometryPart> {
        let offset: u64 = fp.stream_position().unwrap_or(0);
        let part_line: String = read_line(fp)?;
        if !part_line.starts_with("part") {
            return Err(EnsightReaderError::new("Expected 'part' line"));
        }
        if changing_geometry_per_part {
            let _ = part_line;
        }
        let part_id: i32 = read_i32(fp)?;
        let part_name: String = read_line(fp)?.trim_end().to_string();
        let coordinates_line: String = read_line(fp)?;
        if !coordinates_line.starts_with("coordinates") {
            return Err(EnsightReaderError::new("Expected 'coordinates' line"));
        }
        let number_of_nodes: usize = read_i32(fp)? as usize;

        if node_id_handling.ids_present() {
            skip_bytes(fp, number_of_nodes * SIZE_INT)?;
        }
        skip_bytes(fp, 3 * number_of_nodes * SIZE_FLOAT)?;

        let file_len: u64 = file_len(fp)?;
        let mut element_blocks: Vec<UnstructuredElementBlock> = Vec::new();
        while fp.stream_position().unwrap_or(0) < file_len {
            let element_type_line: String = peek_line(fp)?;
            if element_type_line.starts_with("part") {
                break;
            }
            let block: UnstructuredElementBlock =
                UnstructuredElementBlock::from_file(fp, element_id_handling, part_id)?;
            element_blocks.push(block);
        }

        Ok(GeometryPart {
            offset,
            part_id,
            part_name,
            number_of_nodes,
            element_blocks,
            node_id_handling,
            element_id_handling,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VariableArray {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct EnsightVariableFile {
    pub file_path: PathBuf,
    pub variable_name: String,
    pub variable_location: VariableLocation,
    pub variable_type: VariableType,
    pub part_offsets: HashMap<i32, u64>,
    pub geometry_file: EnsightGeometryFile,
}

impl EnsightVariableFile {
    pub fn from_file_path(
        path: impl AsRef<Path>,
        variable_name: String,
        variable_location: VariableLocation,
        variable_type: VariableType,
        geofile: EnsightGeometryFile,
    ) -> Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let mut fp: BufReader<File> = open_reader(&path)?;
        let _ = read_line(&mut fp)?; // description

        let mut part_offsets: HashMap<i32, u64> = HashMap::new();
        let file_len: u64 = file_len(&mut fp)?;
        while fp.stream_position().unwrap_or(0) < file_len {
            let part_offset: u64 = fp.stream_position().unwrap_or(0);
            let part_line: String = read_line(&mut fp)?;
            if !part_line.starts_with("part") {
                return Err(EnsightReaderError::new(format!(
                    "Expected 'part' line, got: {part_line}"
                )));
            }
            let part_id: i32 = read_i32(&mut fp)?;
            let part: &GeometryPart = geofile.get_part_by_id(part_id).ok_or_else(|| {
                EnsightReaderError::new(format!("Part id {part_id} not found in geometry"))
            })?;
            if part_offsets.contains_key(&part_id) {
                return Err(EnsightReaderError::new(format!(
                    "Duplicate definition of part id {part_id}"
                )));
            }

            match variable_location {
                VariableLocation::PerNode => {
                    let coordinates_line = read_line(&mut fp)?;
                    if !coordinates_line.starts_with("coordinates") {
                        return Err(EnsightReaderError::new(format!(
                            "Expected 'coordinates' line, got: {coordinates_line}"
                        )));
                    }
                    if coordinates_line.contains("undef") {
                        let _ = read_f32(&mut fp)?;
                    } else if coordinates_line.contains("partial") {
                        return Err(EnsightReaderError::new(
                            "'coordinates partial' is not supported",
                        ));
                    }
                    let n: usize = part.number_of_nodes;
                    let k: usize = variable_type.value_count();
                    skip_bytes(&mut fp, n * k * SIZE_FLOAT)?;
                    part_offsets.insert(part_id, part_offset);
                }
                VariableLocation::PerElement => {
                    return Err(EnsightReaderError::new(
                        "Per-element variables are not implemented",
                    ));
                }
            }
        }

        Ok(Self {
            file_path: path,
            variable_name,
            variable_location,
            variable_type,
            part_offsets,
            geometry_file: geofile,
        })
    }

    pub fn open(&self) -> Result<Reader> {
        open_reader(&self.file_path)
    }

    pub fn read_node_data(&self, fp: &mut Reader, part_id: i32) -> Result<Option<VariableArray>> {
        if self.variable_location != VariableLocation::PerNode {
            return Err(EnsightReaderError::new("Variable is not per node"));
        }
        let offset: u64 = match self.part_offsets.get(&part_id) {
            Some(v) => *v,
            None => return Ok(None),
        };
        fp.seek(SeekFrom::Start(offset))
            .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
        let part_line: String = read_line(fp)?;
        if !part_line.starts_with("part") {
            return Err(EnsightReaderError::new("Expected 'part' line"));
        }
        let part_id_read: i32 = read_i32(fp)?;
        if part_id_read != part_id {
            return Err(EnsightReaderError::new("Part id mismatch"));
        }
        let coordinates_line: String = read_line(fp)?;
        if !coordinates_line.starts_with("coordinates") {
            return Err(EnsightReaderError::new("Expected 'coordinates' line"));
        }
        if coordinates_line.contains("undef") {
            let _ = read_f32(fp)?;
        }
        let part: &GeometryPart = self.geometry_file.get_part_by_id(part_id).ok_or_else(|| {
            EnsightReaderError::new(format!("Part id {part_id} not found in geometry"))
        })?;
        let n: usize = part.number_of_nodes;
        let k: usize = self.variable_type.value_count();
        let data: Vec<f32> = read_f32_vec(fp, n * k)?;
        Ok(Some(VariableArray {
            data,
            rows: n,
            cols: k,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct EnsightGeometryFileSet {
    pub casefile_dir_path: PathBuf,
    pub timeset: Option<Timeset>,
    pub filename: String,
    pub changing_geometry_per_part: bool,
}

impl EnsightGeometryFileSet {
    pub fn get_file(&self, timestep: usize) -> Result<EnsightGeometryFile> {
        let filename: String = match &self.timeset {
            None => self.filename.clone(),
            Some(ts) => fill_wildcard(&self.filename, ts.filename_numbers[timestep]),
        };
        let path: PathBuf = self.casefile_dir_path.join(filename);
        EnsightGeometryFile::from_file_path(path, self.changing_geometry_per_part)
    }
}

#[derive(Debug, Clone)]
pub struct EnsightVariableFileSet {
    pub casefile_dir_path: PathBuf,
    pub timeset: Option<Timeset>,
    pub variable_location: VariableLocation,
    pub variable_type: VariableType,
    pub variable_name: String,
    pub filename: String,
    pub geometry_model: EnsightGeometryFileSet,
}

impl EnsightVariableFileSet {
    pub fn get_file(&self, timestep: usize) -> Result<EnsightVariableFile> {
        let geofile: EnsightGeometryFile = self.geometry_model.get_file(timestep)?;
        let filename: String = match &self.timeset {
            None => self.filename.clone(),
            Some(ts) => fill_wildcard(&self.filename, ts.filename_numbers[timestep]),
        };
        let path: PathBuf = self.casefile_dir_path.join(filename);
        EnsightVariableFile::from_file_path(
            path,
            self.variable_name.clone(),
            self.variable_location,
            self.variable_type,
            geofile,
        )
    }
}

#[derive(Debug, Clone)]
pub struct EnsightCaseFile {
    pub casefile_path: PathBuf,
    pub geometry_model: EnsightGeometryFileSet,
    pub variables: HashMap<String, EnsightVariableFileSet>,
    pub timesets: HashMap<i32, Timeset>,
}

impl EnsightCaseFile {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let casefile_dir_path: PathBuf = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut fp: BufReader<File> = std::io::BufReader::new(
            File::open(&path)
                .map_err(|e| EnsightReaderError::new(format!("Failed to open case file: {e}")))?,
        );

        let mut geometry_model: Option<EnsightGeometryFileSet> = None;
        let mut geometry_model_ts: Option<i32> = None;
        let mut variables: HashMap<String, EnsightVariableFileSet> = HashMap::new();
        let mut variables_ts: HashMap<String, Option<i32>> = HashMap::new();
        let mut timesets: HashMap<i32, Timeset> = HashMap::new();

        let mut current_section: Option<String> = None;
        let mut current_timeset: Option<Timeset> = None;
        let mut current_timeset_file_start_number: Option<i32> = None;
        let mut current_timeset_filename_increment: Option<i32> = None;
        let mut changing_geometry_per_part = false;
        let mut last_key: Option<String> = None;

        let mut lineno: usize = 0usize;
        let mut line: String = String::new();
        loop {
            line.clear();
            let bytes: usize = fp.read_line(&mut line).map_err(|e: std::io::Error| {
                EnsightReaderError::new(format!("Failed to read case file: {e}"))
            })?;
            if bytes == 0 {
                break;
            }
            lineno += 1;
            let raw: &str = line.trim();
            if raw.is_empty() || raw.starts_with('#') {
                continue;
            }
            if raw
                .chars()
                .all(|c: char| c.is_ascii_uppercase() || c == ' ')
            {
                current_section = Some(raw.to_string());
                continue;
            }
            let (key, values) = if let Some((k, v)) = raw.split_once(':') {
                let values: Vec<String> = v
                    .split_whitespace()
                    .map(|s: &str| s.to_string())
                    .collect::<Vec<_>>();
                last_key = Some(k.trim().to_string());
                (Some(k.trim().to_string()), values)
            } else {
                (
                    None,
                    raw.split_whitespace()
                        .map(|s: &str| s.to_string())
                        .collect::<Vec<_>>(),
                )
            };

            match current_section.as_deref() {
                Some("FORMAT") => {
                    if let Some(k) = key.as_deref() {
                        if k == "type" && values != ["ensight", "gold"] {
                            return Err(EnsightReaderError::with_context(
                                "Expected 'ensight gold' in type line",
                                Some(&path),
                                None,
                                Some(lineno),
                            ));
                        }
                    }
                }
                Some("GEOMETRY") => {
                    if let Some(k) = key.as_deref() {
                        if k == "model" {
                            if values
                                .last()
                                .map(|v: &String| v == "changing_geometry_per_part")
                                .unwrap_or(false)
                            {
                                changing_geometry_per_part = true;
                            }
                            let mut values: Vec<String> = values.clone();
                            if values
                                .last()
                                .map(|v: &String| v == "changing_geometry_per_part")
                                .unwrap_or(false)
                            {
                                values.pop();
                            }
                            if values.len() == 1 {
                                let filename: String = strip_quotes(&values[0]);
                                geometry_model = Some(EnsightGeometryFileSet {
                                    casefile_dir_path: casefile_dir_path.clone(),
                                    timeset: None,
                                    filename,
                                    changing_geometry_per_part,
                                });
                            } else if values.len() == 2 {
                                geometry_model_ts = values[0].parse::<i32>().ok();
                                let filename: String = strip_quotes(&values[1]);
                                geometry_model = Some(EnsightGeometryFileSet {
                                    casefile_dir_path: casefile_dir_path.clone(),
                                    timeset: None,
                                    filename,
                                    changing_geometry_per_part,
                                });
                            } else {
                                return Err(EnsightReaderError::with_context(
                                    "Unsupported model definition",
                                    Some(&path),
                                    None,
                                    Some(lineno),
                                ));
                            }
                        }
                    }
                }
                Some("VARIABLE") => {
                    if let Some(k) = key.as_deref() {
                        let parts: Vec<&str> = k.split(" per ").collect();
                        if parts.len() == 2 {
                            let variable_type: VariableType = VariableType::from_str(parts[0])?;
                            let variable_location: VariableLocation =
                                VariableLocation::from_str(parts[1])?;

                            if values.len() < 2 {
                                return Err(EnsightReaderError::new("Unsupported variable line"));
                            }
                            let filename: String = strip_quotes(values.last().unwrap());
                            let description: String = values
                                .get(values.len() - 2)
                                .cloned()
                                .unwrap_or_else(|| "var".to_string());
                            let ts: Option<i32> = if values.len() == 3 {
                                values[0].parse::<i32>().ok()
                            } else {
                                None
                            };

                            variables.insert(
                                description.clone(),
                                EnsightVariableFileSet {
                                    casefile_dir_path: casefile_dir_path.clone(),
                                    timeset: None,
                                    variable_location,
                                    variable_type,
                                    variable_name: description.clone(),
                                    filename,
                                    geometry_model: EnsightGeometryFileSet {
                                        casefile_dir_path: PathBuf::new(),
                                        timeset: None,
                                        filename: String::new(),
                                        changing_geometry_per_part,
                                    },
                                },
                            );
                            variables_ts.insert(description, ts);
                        }
                    }
                }
                Some("TIME") => {
                    if let Some(k) = key.as_deref() {
                        match k {
                            "time set" => {
                                let ts: i32 = values
                                    .get(0)
                                    .and_then(|v: &String| v.parse::<i32>().ok())
                                    .unwrap_or(0);
                                let description: Option<String> = values.get(1).cloned();
                                let timeset: Timeset = Timeset {
                                    timeset_id: ts,
                                    description,
                                    number_of_steps: 0,
                                    filename_numbers: Vec::new(),
                                    time_values: Vec::new(),
                                };
                                timesets.insert(ts, timeset.clone());
                                current_timeset = Some(timeset);
                                current_timeset_file_start_number = None;
                                current_timeset_filename_increment = None;
                            }
                            "number of steps" => {
                                if let Some(ts) = current_timeset.as_mut() {
                                    ts.number_of_steps = values
                                        .get(0)
                                        .and_then(|v| v.parse::<usize>().ok())
                                        .unwrap_or(0);
                                }
                            }
                            "filename start number" => {
                                current_timeset_file_start_number =
                                    values.get(0).and_then(|v| v.parse::<i32>().ok());
                                if let (Some(ts), Some(start), Some(inc)) = (
                                    current_timeset.as_mut(),
                                    current_timeset_file_start_number,
                                    current_timeset_filename_increment,
                                ) {
                                    ts.filename_numbers =
                                        Timeset::filename_numbers_from_arithmetic_sequence(
                                            start,
                                            ts.number_of_steps,
                                            inc,
                                        );
                                }
                            }
                            "filename increment" => {
                                current_timeset_filename_increment =
                                    values.get(0).and_then(|v| v.parse::<i32>().ok());
                                if let (Some(ts), Some(start), Some(inc)) = (
                                    current_timeset.as_mut(),
                                    current_timeset_file_start_number,
                                    current_timeset_filename_increment,
                                ) {
                                    ts.filename_numbers =
                                        Timeset::filename_numbers_from_arithmetic_sequence(
                                            start,
                                            ts.number_of_steps,
                                            inc,
                                        );
                                }
                            }
                            "filename numbers" => {
                                if let Some(ts) = current_timeset.as_mut() {
                                    ts.filename_numbers.extend(
                                        values
                                            .iter()
                                            .filter_map(|v: &String| v.parse::<i32>().ok()),
                                    );
                                }
                            }
                            "time values" => {
                                if let Some(ts) = current_timeset.as_mut() {
                                    ts.time_values.extend(
                                        values
                                            .iter()
                                            .filter_map(|v: &String| v.parse::<f64>().ok()),
                                    );
                                }
                            }
                            "filename numbers file" => {
                                if let Some(ts) = current_timeset.as_mut() {
                                    let path: PathBuf = casefile_dir_path.join(strip_quotes(
                                        values.get(0).unwrap_or(&"".to_string()),
                                    ));
                                    ts.filename_numbers =
                                        read_numbers_from_text_file(&path, |s: &str| {
                                            s.parse::<i32>().ok()
                                        })?;
                                }
                            }
                            "time values file" => {
                                if let Some(ts) = current_timeset.as_mut() {
                                    let path: PathBuf = casefile_dir_path.join(strip_quotes(
                                        values.get(0).unwrap_or(&"".to_string()),
                                    ));
                                    ts.time_values =
                                        read_numbers_from_text_file(&path, |s: &str| {
                                            s.parse::<f64>().ok()
                                        })?;
                                }
                            }
                            _ => {}
                        }
                    } else if let Some(last) = last_key.as_deref() {
                        if last == "time values" {
                            if let Some(ts) = current_timeset.as_mut() {
                                ts.time_values.extend(
                                    values.iter().filter_map(|v: &String| v.parse::<f64>().ok()),
                                );
                            }
                        } else if last == "filename numbers" {
                            if let Some(ts) = current_timeset.as_mut() {
                                ts.filename_numbers.extend(
                                    values.iter().filter_map(|v: &String| v.parse::<i32>().ok()),
                                );
                            }
                        }
                    }
                    if let Some(ts) = current_timeset.as_ref() {
                        timesets.insert(ts.timeset_id, ts.clone());
                    }
                }
                _ => {}
            }
        }

        let mut geometry_model: EnsightGeometryFileSet = geometry_model
            .ok_or_else(|| EnsightReaderError::new("No model defined in casefile"))?;
        let default_ts: Option<i32> = timesets.keys().min().copied();
        if geometry_model_ts.is_none() && geometry_model.filename.contains('*') {
            geometry_model_ts = default_ts;
        }
        if let Some(ts_id) = geometry_model_ts {
            if let Some(ts) = timesets.get(&ts_id) {
                geometry_model.timeset = Some(ts.clone());
            }
        }
        for (name, ts) in variables_ts.iter() {
            if let Some(var) = variables.get_mut(name) {
                if let Some(ts_id) = ts.or(default_ts) {
                    if let Some(ts_data) = timesets.get(&ts_id) {
                        var.timeset = Some(ts_data.clone());
                    }
                }
            }
        }
        for var in variables.values_mut() {
            var.geometry_model = geometry_model.clone();
        }

        Ok(Self {
            casefile_path: path,
            geometry_model,
            variables,
            timesets,
        })
    }

    pub fn get_geometry_model(&self, timestep: usize) -> Result<EnsightGeometryFile> {
        self.geometry_model.get_file(timestep)
    }

    pub fn get_variable(&self, name: &str, timestep: usize) -> Result<EnsightVariableFile> {
        let var: &EnsightVariableFileSet = self
            .variables
            .get(name)
            .ok_or_else(|| EnsightReaderError::new(format!("Variable not found: {name}")))?;
        var.get_file(timestep)
    }
}

pub fn read_case(path: impl AsRef<Path>) -> Result<EnsightCaseFile> {
    EnsightCaseFile::from_file(path)
}

pub type Reader = BufReader<File>;

fn open_reader(path: &Path) -> Result<Reader> {
    let file: File = File::open(path).map_err(|e: std::io::Error| {
        EnsightReaderError::new(format!("Failed to open file: {e}"))
    })?;
    Ok(BufReader::new(file))
}

fn read_line(fp: &mut Reader) -> Result<String> {
    let mut buf: Vec<u8> = vec![0u8; LINE_LEN];
    fp.read_exact(&mut buf).map_err(|e: std::io::Error| {
        EnsightReaderError::new(format!("Failed to read line: {e}"))
    })?;
    let line: String = String::from_utf8_lossy(&buf)
        .trim_end_matches(|c: char| c == '\0' || c == ' ')
        .to_string();
    Ok(line)
}

fn peek_line(fp: &mut Reader) -> Result<String> {
    let pos: u64 = fp
        .stream_position()
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to tell: {e}")))?;
    let line: String = read_line(fp)?;
    fp.seek(SeekFrom::Start(pos))
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek: {e}")))?;
    Ok(line)
}

fn read_i32(fp: &mut Reader) -> Result<i32> {
    fp.read_i32::<LittleEndian>()
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to read i32: {e}")))
}

fn read_f32(fp: &mut Reader) -> Result<f32> {
    fp.read_f32::<LittleEndian>()
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to read f32: {e}")))
}

fn read_i32_vec(fp: &mut Reader, count: usize) -> Result<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_i32(fp)?);
    }
    Ok(out)
}

fn read_f32_vec(fp: &mut Reader, count: usize) -> Result<Vec<f32>> {
    let mut out: Vec<f32> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_f32(fp)?);
    }
    Ok(out)
}

fn skip_bytes(fp: &mut Reader, count: usize) -> Result<()> {
    fp.seek(SeekFrom::Current(count as i64))
        .map_err(|e: std::io::Error| {
            EnsightReaderError::new(format!("Failed to skip bytes: {e}"))
        })?;
    Ok(())
}

fn file_len(fp: &mut Reader) -> Result<u64> {
    let pos: u64 = fp
        .stream_position()
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to tell: {e}")))?;
    let len: u64 = fp
        .seek(SeekFrom::End(0))
        .map_err(|e: std::io::Error| EnsightReaderError::new(format!("Failed to seek end: {e}")))?;
    fp.seek(SeekFrom::Start(pos)).map_err(|e: std::io::Error| {
        EnsightReaderError::new(format!("Failed to seek back: {e}"))
    })?;
    Ok(len)
}

fn parse_id_handling(line: &str, prefix: &str) -> Result<IdHandling> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 3 && line.starts_with(prefix) {
        IdHandling::from_str(parts[parts.len() - 1])
    } else {
        Err(EnsightReaderError::new(format!(
            "Unexpected '{prefix}' line: {line}"
        )))
    }
}

fn fill_wildcard(filename: &str, value: i32) -> String {
    if let Some(pos) = filename.find('*') {
        let run: usize = filename[pos..]
            .chars()
            .take_while(|c: &char| *c == '*')
            .count();
        let pad: String = format!("{:0width$}", value, width = run);
        let mut out: String = filename.to_string();
        out.replace_range(pos..pos + run, &pad);
        out
    } else {
        filename.to_string()
    }
}

fn strip_quotes(s: &str) -> String {
    s.trim_matches('"').to_string()
}

fn read_numbers_from_text_file<T, F>(path: &Path, parser: F) -> Result<Vec<T>>
where
    F: Fn(&str) -> Option<T>,
{
    let content: String = std::fs::read_to_string(path).map_err(|e: std::io::Error| {
        EnsightReaderError::new(format!("Failed to read file {path:?}: {e}"))
    })?;
    let mut out: Vec<T> = Vec::new();
    for token in content.split_whitespace() {
        if let Some(val) = parser(token) {
            out.push(val);
        }
    }
    Ok(out)
}
