#![allow(dead_code)]
use half;
use safetensors::tensor::{Dtype, View};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::PartialOrd;
use std::collections::HashMap;
use std::default::Default;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum DataType {
    BOOL,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::BOOL => 1,
            DataType::U8 => 1,
            DataType::I8 => 1,
            DataType::I16 => 2,
            DataType::U16 => 2,
            DataType::I32 => 4,
            DataType::U32 => 4,
            DataType::I64 => 8,
            DataType::U64 => 8,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::F32 => 4,
            DataType::F64 => 8,
        }
    }
}

impl Into<Dtype> for DataType {
    fn into(self) -> Dtype {
        match self {
            DataType::BOOL => Dtype::BOOL,
            DataType::U8 => Dtype::U8,
            DataType::I8 => Dtype::I8,
            DataType::I16 => Dtype::I16,
            DataType::U16 => Dtype::U16,
            DataType::F16 => Dtype::F16,
            DataType::BF16 => Dtype::BF16,
            DataType::I32 => Dtype::I32,
            DataType::U32 => Dtype::U32,
            DataType::F32 => Dtype::F32,
            DataType::F64 => Dtype::F64,
            DataType::I64 => Dtype::I64,
            DataType::U64 => Dtype::U64,
        }
    }
}

trait DataTypeTrait {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self
    where
        Self: Sized;
}

impl DataTypeTrait for bool {
    fn to_bytes(&self) -> Vec<u8> {
        if *self {
            vec![1]
        } else {
            vec![0]
        }
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] == 1
    }
}

impl DataTypeTrait for u8 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        u8::from_le_bytes([bytes[0]])
    }
}

impl DataTypeTrait for i8 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i8::from_le_bytes([bytes[0]])
    }
}

impl DataTypeTrait for i16 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i16::from_le_bytes([bytes[0], bytes[1]])
    }
}

impl DataTypeTrait for u16 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        u16::from_le_bytes([bytes[0], bytes[1]])
    }
}

impl DataTypeTrait for i32 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl DataTypeTrait for u32 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl DataTypeTrait for i64 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

impl DataTypeTrait for u64 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

impl DataTypeTrait for half::f16 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        half::f16::from_le_bytes([bytes[0], bytes[1]])
    }
}

impl DataTypeTrait for half::bf16 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        half::bf16::from_le_bytes([bytes[0], bytes[1]])
    }
}

impl DataTypeTrait for f32 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl DataTypeTrait for f64 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

struct Tensor<T: DataTypeTrait> {
    data: Vec<T>,
    shape: Vec<usize>,
    dtype: DataType,
}

impl<T: DataTypeTrait + Default + Clone + Copy> Tensor<T> {
    fn new(data: Vec<T>, shape: Vec<usize>, dtype: DataType) -> Self {
        Tensor { data, shape, dtype }
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        for d in &self.data {
            bytes.extend(d.to_bytes());
        }
        bytes
    }
    fn from_bytes(bytes: &[u8], shape: Vec<usize>, dtype: DataType) -> Self {
        let mut data = vec![];
        let size = shape.iter().product();
        for i in 0..size {
            let start = i * dtype.size();
            let end = start + dtype.size();
            data.push(T::from_bytes(&bytes[start..end]));
        }
        Tensor { data, shape, dtype }
    }
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn strides(&self) -> Vec<usize> {
        let mut strides = vec![0; self.shape.len()];
        let mut stride = 1;
        for i in (0..strides.len()).rev() {
            strides[i] = stride;
            stride *= self.shape[i];
        }
        strides
    }
    fn strides_from_shape(shape: Vec<usize>) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..strides.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }
    fn permute(&self, dims: Vec<usize>) -> Self {
        let mut data = vec![T::default(); self.numel()];

        let shape = dims.iter().map(|&i| self.shape[i]).collect::<Vec<usize>>();
        let strides = self.strides();
        let new_strides = Self::strides_from_shape(shape.clone());

        for index in 0..self.numel() {
            let mut old_idx = 0;
            let mut temp = index;
            for (&dim, &stride) in dims.iter().zip(new_strides.iter()) {
                old_idx += (temp / stride) * strides[dim];
                temp %= stride;
            }
            data[index] = self.data[old_idx];
        }

        Tensor {
            data,
            shape,
            dtype: self.dtype,
        }
    }
    fn view(&self) -> SafeTensor {
        SafeTensor {
            dtype: self.dtype,
            shape: self.shape.clone(),
            data: self.to_bytes(),
        }
    }
}

struct Functional;

impl Functional {
    pub fn arange(start: f32, end: f32, step: f32) -> Tensor<f32> {
        let seqlen = ((end - start) / step).ceil() as usize;
        let mut data = vec![];
        let mut i = start;
        while i < end {
            data.push(i);
            i += step;
        }
        Tensor::new(data, vec![seqlen], DataType::F32)
    }
}

struct To<T, T2> {
    _t: std::marker::PhantomData<(T, T2)>,
}

impl To<f32, i64> {
    fn tensor(tensor: &Tensor<f32>) -> Tensor<i64> {
        let data = tensor.data.iter().map(|&x| x as i64).collect();
        Tensor::new(data, tensor.shape.clone(), DataType::I64)
    }
}

impl To<i64, f32> {
    fn tensor(tensor: &Tensor<i64>) -> Tensor<f32> {
        let data = tensor.data.iter().map(|&x| x as f32).collect();
        Tensor::new(data, tensor.shape.clone(), DataType::F32)
    }
}

struct SafeTensor {
    dtype: DataType,
    shape: Vec<usize>,
    data: Vec<u8>,
}
impl<'data> View for &'data SafeTensor {
    fn dtype(&self) -> Dtype {
        self.dtype.into()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<[u8]> {
        (&self.data).into()
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

struct Model {
    weights: HashMap<String, SafeTensor>,
}

impl Model {
    fn new() -> Self {
        Model {
            weights: HashMap::new(),
        }
    }
    fn insert(&mut self, key: &str, value: SafeTensor) {
        self.weights.insert(key.to_string(), value);
    }
    fn get(&self, key: &str) -> Option<&SafeTensor> {
        self.weights.get(key)
    }
    fn save(&self, path: &Path) {
        let model_dict: Vec<(&str, &SafeTensor)> =
            self.weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
        safetensors::serialize_to_file(model_dict, &None, path).expect("Failed to save model");
    }
}
fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = vec![1, 2, 4];
    let dtype = DataType::F32;
    let tensor: Tensor<f32> = Tensor::new(data, shape.clone(), dtype);
    let bytes = tensor.to_bytes();
    let tensor2: Tensor<f32> = Tensor::from_bytes(&bytes, shape, dtype);
    println!("{:?}", tensor2.data);
    println!("{:?}", tensor2.shape);
    let tensor3: Tensor<f32> = tensor2.permute(vec![0, 2, 1]);
    println!("{:?}", tensor3.data);
    println!("{:?}", tensor3.shape);
    let tensor4 = Functional::arange(0.0, 10.0, 1.0);
    println!("{:?}", tensor4.data);
    println!("{:?}", tensor4.shape);
    let tensor5: Tensor<i64> = To::<f32, i64>::tensor(&tensor4);
    println!("{:?}", tensor5.data);
    println!("{:?}", tensor5.shape);
    println!("{:?}", tensor5.dtype);

    let mut model = Model::new();
    model.insert("weights", tensor3.view());
    model.insert("weights2", tensor4.view());
    let safe_tensor = model.get("weights").unwrap();
    println!("{:?}", safe_tensor.shape());

    model.save(Path::new("model.safetensors"));
}
