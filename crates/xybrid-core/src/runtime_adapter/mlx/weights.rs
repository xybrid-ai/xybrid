//! SafeTensors bundle resolution for MLX model loaders.
//!
//! MLX-LM publishes both single-file bundles (`model.safetensors`) and
//! HuggingFace-indexed shards (`model.safetensors.index.json` plus
//! `model-00001-of-00002.safetensors`, ...). The architecture builders
//! consume this module so validation and runtime materialisation share
//! one view of the bundle layout.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

use safetensors::SafeTensors;
use serde::Deserialize;

use super::model::{MlxLlmError, MlxLlmResult};

#[derive(Debug)]
pub struct SafeTensorBundle {
    layout: SafeTensorLayout,
    file_cache: RefCell<Option<(PathBuf, Rc<Vec<u8>>)>>,
}

#[derive(Debug)]
enum SafeTensorLayout {
    Single {
        path: PathBuf,
    },
    Sharded {
        index_path: PathBuf,
        weight_map: HashMap<String, PathBuf>,
    },
}

#[derive(Debug, Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

impl SafeTensorBundle {
    pub fn from_model_dir(dir: &Path) -> MlxLlmResult<Self> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Ok(Self::from_single_file(single));
        }

        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return Self::from_index(dir, index_path);
        }

        if let Some(shard) = first_safetensors_shard(dir) {
            return Err(MlxLlmError::OrphanShardedWeights {
                marker: shard,
                dir: dir.to_path_buf(),
            });
        }

        Err(MlxLlmError::MissingFile {
            file: "model.safetensors",
            dir: dir.to_path_buf(),
        })
    }

    pub fn from_single_file(path: PathBuf) -> Self {
        Self {
            layout: SafeTensorLayout::Single { path },
            file_cache: RefCell::new(None),
        }
    }

    fn from_index(dir: &Path, index_path: PathBuf) -> MlxLlmResult<Self> {
        let raw = fs::read_to_string(&index_path).map_err(|e| MlxLlmError::WeightLoad {
            path: index_path.clone(),
            reason: format!("read safetensors index: {e}"),
        })?;
        let index: SafeTensorIndex =
            serde_json::from_str(&raw).map_err(|e| MlxLlmError::WeightLoad {
                path: index_path.clone(),
                reason: format!("parse safetensors index: {e}"),
            })?;
        if index.weight_map.is_empty() {
            return Err(MlxLlmError::WeightLoad {
                path: index_path,
                reason: "safetensors index has an empty weight_map".into(),
            });
        }

        let mut weight_map = HashMap::with_capacity(index.weight_map.len());
        for (tensor, file) in index.weight_map {
            let shard = checked_shard_path(dir, &file, &index_path)?;
            weight_map.insert(tensor, shard);
        }

        Ok(Self {
            layout: SafeTensorLayout::Sharded {
                index_path,
                weight_map,
            },
            file_cache: RefCell::new(None),
        })
    }

    pub fn tensor_names(&self) -> MlxLlmResult<HashSet<String>> {
        match &self.layout {
            SafeTensorLayout::Single { path } => tensor_names_from_file(self, path),
            SafeTensorLayout::Sharded { weight_map, .. } => {
                let mut names = HashSet::new();
                for path in unique_paths(weight_map.values()) {
                    names.extend(tensor_names_from_file(self, &path)?);
                }
                Ok(names)
            }
        }
    }

    pub fn path_for_error(&self) -> PathBuf {
        match &self.layout {
            SafeTensorLayout::Single { path } => path.clone(),
            SafeTensorLayout::Sharded { index_path, .. } => index_path.clone(),
        }
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub fn read_as_f32(&self, name: &str) -> MlxLlmResult<(Vec<f32>, Vec<i32>)> {
        use safetensors::Dtype as StDtype;
        use xybrid_mlx::{bf16_le_bytes_to_f32_vec, f16_le_bytes_to_f32_vec};

        let path = self.tensor_path(name)?;
        let bytes = self.file_bytes(&path)?;
        let st =
            SafeTensors::deserialize(bytes.as_slice()).map_err(|e| MlxLlmError::WeightLoad {
                path: path.clone(),
                reason: format!("parse safetensors: {e}"),
            })?;
        let view = st.tensor(name).map_err(|e| MlxLlmError::WeightLoad {
            path: path.clone(),
            reason: format!("tensor `{name}` missing: {e}"),
        })?;
        let shape_i32 = safetensor_shape_to_i32(view.shape(), &path, name)?;
        let data = view.data();

        let floats: Vec<f32> = match view.dtype() {
            StDtype::F32 => {
                f32_le_bytes_to_vec(data).map_err(|reason| MlxLlmError::WeightLoad {
                    path: path.clone(),
                    reason,
                })?
            }
            StDtype::F16 => f16_le_bytes_to_f32_vec(data)?,
            StDtype::BF16 => bf16_le_bytes_to_f32_vec(data)?,
            other => {
                return Err(MlxLlmError::WeightLoad {
                    path,
                    reason: format!("unsupported tensor dtype {other:?}"),
                });
            }
        };
        Ok((floats, shape_i32))
    }

    #[cfg(all(
        feature = "llm-mlx-runtime",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub fn read_array(&self, name: &str) -> MlxLlmResult<xybrid_mlx::MlxArray> {
        use safetensors::Dtype as StDtype;
        use xybrid_mlx::{MlxArray, MlxDtype};

        let path = self.tensor_path(name)?;
        let bytes = self.file_bytes(&path)?;
        let st =
            SafeTensors::deserialize(bytes.as_slice()).map_err(|e| MlxLlmError::WeightLoad {
                path: path.clone(),
                reason: format!("parse safetensors: {e}"),
            })?;
        let view = st.tensor(name).map_err(|e| MlxLlmError::WeightLoad {
            path: path.clone(),
            reason: format!("tensor `{name}` missing: {e}"),
        })?;
        let shape_i32 = safetensor_shape_to_i32(view.shape(), &path, name)?;
        let dtype = match view.dtype() {
            StDtype::F32 => MlxDtype::F32,
            StDtype::F16 => MlxDtype::F16,
            StDtype::BF16 => MlxDtype::Bf16,
            other => {
                return Err(MlxLlmError::WeightLoad {
                    path,
                    reason: format!("unsupported tensor dtype {other:?}"),
                });
            }
        };
        MlxArray::from_dtype_bytes(view.data(), &shape_i32, dtype).map_err(Into::into)
    }

    fn tensor_path(&self, name: &str) -> MlxLlmResult<PathBuf> {
        match &self.layout {
            SafeTensorLayout::Single { path } => Ok(path.clone()),
            SafeTensorLayout::Sharded {
                index_path,
                weight_map,
            } => weight_map
                .get(name)
                .cloned()
                .ok_or_else(|| MlxLlmError::WeightLoad {
                    path: index_path.clone(),
                    reason: format!("safetensors index missing tensor `{name}`"),
                }),
        }
    }

    fn file_bytes(&self, path: &Path) -> MlxLlmResult<Rc<Vec<u8>>> {
        if let Some((cached_path, bytes)) = self.file_cache.borrow().as_ref() {
            if cached_path == path {
                return Ok(bytes.clone());
            }
        }
        let bytes = Rc::new(fs::read(path).map_err(|e| MlxLlmError::WeightLoad {
            path: path.to_path_buf(),
            reason: format!("read safetensors: {e}"),
        })?);
        *self.file_cache.borrow_mut() = Some((path.to_path_buf(), bytes.clone()));
        Ok(bytes)
    }
}

fn f32_le_bytes_to_vec(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(4) {
        return Err(format!(
            "f32 tensor byte length {} is not divisible by 4",
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn safetensor_shape_to_i32(shape: &[usize], path: &Path, name: &str) -> MlxLlmResult<Vec<i32>> {
    let mut out = Vec::with_capacity(shape.len());
    for (axis, &dim) in shape.iter().enumerate() {
        let dim = i32::try_from(dim).map_err(|_| MlxLlmError::WeightLoad {
            path: path.to_path_buf(),
            reason: format!("tensor `{name}` dimension {axis}={dim} exceeds MLX i32 shape limit"),
        })?;
        out.push(dim);
    }
    Ok(out)
}

fn tensor_names_from_file(bundle: &SafeTensorBundle, path: &Path) -> MlxLlmResult<HashSet<String>> {
    let bytes = bundle.file_bytes(path)?;
    let (_, meta) =
        SafeTensors::read_metadata(bytes.as_slice()).map_err(|e| MlxLlmError::WeightLoad {
            path: path.to_path_buf(),
            reason: format!("invalid safetensors header: {e}"),
        })?;
    Ok(meta.tensors().into_keys().collect())
}

fn unique_paths<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for path in paths {
        if seen.insert(path.clone()) {
            unique.push(path.clone());
        }
    }
    unique
}

fn checked_shard_path(dir: &Path, file: &str, index_path: &Path) -> MlxLlmResult<PathBuf> {
    let path = Path::new(file);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(MlxLlmError::WeightLoad {
            path: index_path.to_path_buf(),
            reason: format!("safetensors index references unsafe shard path `{file}`"),
        });
    }
    Ok(dir.join(path))
}

fn first_safetensors_shard(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with("model-") && name.contains("-of-") && name.ends_with(".safetensors") {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::runtime_adapter::mlx::model::MlxLlmError;

    use super::{f32_le_bytes_to_vec, safetensor_shape_to_i32};

    #[test]
    fn f32_decoder_handles_unaligned_slices() {
        let mut bytes = vec![0xAA];
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());

        let decoded = f32_le_bytes_to_vec(&bytes[1..]).unwrap();

        assert_eq!(decoded, vec![1.25, -2.5]);
    }

    #[test]
    fn f32_decoder_rejects_truncated_bytes() {
        let err = f32_le_bytes_to_vec(&[0, 1, 2]).unwrap_err();

        assert!(err.contains("not divisible by 4"), "{err}");
    }

    #[test]
    fn safetensor_shape_rejects_dims_over_mlx_i32_limit() {
        let err = safetensor_shape_to_i32(
            &[2, i32::MAX as usize + 1],
            Path::new("model.safetensors"),
            "model.layers.0.weight",
        )
        .unwrap_err();

        match err {
            MlxLlmError::WeightLoad { reason, .. } => {
                assert!(reason.contains("exceeds MLX i32 shape limit"), "{reason}");
                assert!(reason.contains("dimension 1"), "{reason}");
            }
            other => panic!("expected WeightLoad, got {other:?}"),
        }
    }
}
