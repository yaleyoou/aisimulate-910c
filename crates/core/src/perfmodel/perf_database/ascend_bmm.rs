use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_bmm_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendBmmKey {
    pub role: String,
    pub local_heads: u32,
    pub execution_mode: String,
}

impl AscendBmmKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(
            self.role.as_str(),
            "query_nope" | "output_prefill" | "output_decode"
        ) || !matches!(self.local_heads, 4 | 8 | 16 | 32 | 64)
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend BMM identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn kernel_source(&self) -> &'static str {
        if self.role == "output_decode" {
            "torch.ops.npu.batch_matmul_transpose"
        } else {
            "torch_npu.npu_transpose_batchmatmul"
        }
    }

    pub fn shape(&self) -> (u32, u32, [u32; 3], [u32; 3]) {
        if self.role == "query_nope" {
            (192, 512, [self.local_heads * 256, 256, 1], [98304, 1, 192])
        } else {
            (512, 256, [self.local_heads * 512, 512, 1], [131072, 256, 1])
        }
    }
}

struct Curve {
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
}

type Grids = BTreeMap<AscendBmmKey, Curve>;

pub struct AscendBmmTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendBmmTable {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            grids: OnceLock::new(),
        }
    }

    pub fn with_sources(data_root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, data_root)?
            .into_iter()
            .find(|source| source.channel == "primary");
        if primary
            .as_ref()
            .is_some_and(|source| source.source.kernel_sources().is_some())
        {
            return Err(AicError::InvalidPerfData(
                "Ascend BMM rejects filtered primary sources".into(),
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let family_path = data_root
                .parent()
                .and_then(Path::parent)
                .and_then(|system| {
                    Some(
                        system
                            .join("mla_bmm")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family_path.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend BMM requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendBmmKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend BMM primary data is missing".into()));
                };
                load(path).map_err(|error| (true, error.to_string()))
            })
            .as_ref()
            .map_err(|(invalid, message)| {
                if *invalid {
                    AicError::InvalidPerfData(message.clone())
                } else {
                    AicError::PerfDatabase(message.clone())
                }
            })?;
        grids
            .get(key)
            .ok_or_else(|| AicError::PerfDatabase(format!("unmeasured Ascend BMM tuple: {key:?}")))
    }

    pub fn ensure_key(&self, key: &AscendBmmKey) -> Result<(), AicError> {
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendBmmKey, tokens: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if tokens < curve.minimum || tokens > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend BMM tokens={tokens} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(tokens), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let role = reader.col("role")?;
    let heads = reader.col("local_heads")?;
    let mode = reader.col("execution_mode")?;
    let tokens = reader.col("tokens")?;
    let input_width = reader.col("input_width")?;
    let output_width = reader.col("output_width")?;
    let format = reader.col("format")?;
    let dtype = reader.col("dtype")?;
    let kernel = reader.col("kernel_source")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let verification = reader.col("execution_verification")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let strides = [
        "input_token_stride",
        "input_head_stride",
        "input_inner_stride",
        "weight_head_stride",
        "weight_inner_stride",
        "weight_output_stride",
    ]
    .map(|name| reader.col(name))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let mut raw: BTreeMap<AscendBmmKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identities: BTreeMap<AscendBmmKey, String> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendBmmKey {
            role: row.str_owned(role)?,
            local_heads: row.u32(heads)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let (inner, width, input_stride, weight_stride) = key.shape();
        let strides_valid = strides
            .iter()
            .zip(input_stride.into_iter().chain(weight_stride))
            .try_fold(true, |valid, (column, expected)| {
                Ok::<bool, AicError>(valid && row.u32(*column)? == expected)
            })?;
        let count = row.u32(tokens)?;
        let measured = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let expected_verification = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if !strides_valid
            || row.u32(input_width)? != inner
            || row.u32(output_width)? != width
            || row.u32(format)? != 2
            || row.str(dtype)? != "bfloat16"
            || row.str(kernel)? != key.kernel_source()
            || count == 0
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(verification)? != expected_verification
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend BMM row: {key:?}"
            )));
        }
        let identity = identities
            .entry(key.clone())
            .or_insert_with(|| runtime.to_owned());
        if identity != runtime {
            return Err(AicError::InvalidPerfData(
                "Ascend BMM curve mixes runtime identities".into(),
            ));
        }
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend BMM physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData("empty Ascend BMM table".into()));
    }
    Ok(raw
        .into_iter()
        .map(|(key, points)| {
            (
                key,
                Curve {
                    minimum: *points.first_key_value().unwrap().0,
                    maximum: *points.last_key_value().unwrap().0,
                    values: AxisCurve::from_map("tokens", points),
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::ascend_bmm::AscendBmmOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perf_database::{PerfDbSources, PerfSource};

    fn key() -> AscendBmmKey {
        AscendBmmKey {
            role: "query_nope".into(),
            local_heads: 8,
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("role", vec!["query_nope"; 2]),
            Col::I64("local_heads", vec![8; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("tokens", vec![1, 9]),
            Col::I64("input_width", vec![192; 2]),
            Col::I64("output_width", vec![512; 2]),
            Col::I64("format", vec![2; 2]),
            Col::Str("dtype", vec!["bfloat16"; 2]),
            Col::Str(
                "kernel_source",
                vec!["torch_npu.npu_transpose_batchmatmul"; 2],
            ),
            Col::F64("latency_ms", vec![1.0, 5.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("numerical_checks", vec![24; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
            Col::I64("input_token_stride", vec![2048; 2]),
            Col::I64("input_head_stride", vec![256; 2]),
            Col::I64("input_inner_stride", vec![1; 2]),
            Col::I64("weight_head_stride", vec![98304; 2]),
            Col::I64("weight_inner_stride", vec![1; 2]),
            Col::I64("weight_output_stride", vec![192; 2]),
        ]
    }

    #[test]
    fn measured_tokens_interpolate_without_borrowing_heads_kernel_or_mode() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendBmmTable::new(Some(path));
        assert_eq!(table.query(&key(), 1).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 5).unwrap(), 3.0);
        for tokens in [0, 10] {
            assert!(
                table
                    .query(&key(), tokens)
                    .unwrap_err()
                    .is_missing_perf_data()
            );
        }
        let mut other = key();
        other.local_heads = 16;
        assert!(table.query(&other, 5).unwrap_err().is_missing_perf_data());
        other = key();
        other.execution_mode = "eager".into();
        assert!(table.query(&other, 5).unwrap_err().is_missing_perf_data());
        other = key();
        other.role = "output_decode".into();
        assert!(table.query(&other, 5).unwrap_err().is_missing_perf_data());
    }

    #[test]
    fn wrong_strides_unverified_replays_and_duplicates_are_invalid_data() {
        for (index, replacement) in [
            (3, Col::I64("tokens", vec![1; 2])),
            (10, Col::F64("cv", vec![0.11; 2])),
            (11, Col::I64("numerical_checks", vec![0; 2])),
            (
                13,
                Col::Str("execution_verification", vec!["eager_invocation"; 2]),
            ),
            (16, Col::I64("input_head_stride", vec![192; 2])),
            (19, Col::I64("weight_inner_stride", vec![512; 2])),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            assert!(matches!(
                AscendBmmTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendBmmTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn padding_and_weight_size_use_physical_bf16_views() {
        let mut op = AscendBmmOp {
            name: "query".into(),
            scale_factor: 2.0,
            key: key(),
            token_multiple: 8,
            resident_copies: None,
        };
        assert_eq!(op.measured_tokens(5).unwrap(), 8);
        assert_eq!(op.measured_tokens(9).unwrap(), 16);
        assert_eq!(op.measured_tokens(0).unwrap(), 0);
        assert!(op.measured_tokens(u32::MAX).is_err());
        assert_eq!(op.weight_bytes(), 3_145_728.0);
        op.resident_copies = Some(4);
        assert_eq!(op.weight_bytes(), 6_291_456.0);
        assert_eq!(op.measured_tokens(9).unwrap(), 16);
    }

    #[test]
    fn primary_sources_preserve_hardware_and_runtime_identity() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/ascend910c/mla_bmm/sglang/0.5.22",
            "data/ascend910c/sglang/0.5.22",
        ] {
            let path = root.path().join(relative).join(BASENAME);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            write_parquet(&path, &columns());
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(path, None)],
            )]));
            assert_eq!(
                AscendBmmTable::with_sources(&requested, &resolver)
                    .unwrap()
                    .query(&key(), 5)
                    .unwrap(),
                3.0
            );
        }
        for relative in [
            "data/other/mla_bmm/sglang/0.5.22",
            "data/ascend910c/mla_bmm/vllm/0.5.22",
            "data/ascend910c/mla_bmm/sglang/0.5.21",
        ] {
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(root.path().join(relative).join(BASENAME), None)],
            )]));
            assert!(matches!(
                AscendBmmTable::with_sources(&requested, &resolver),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }
}
