use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::AxisCurve;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_moe_chain_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeChainKey {
    pub phase: String,
    pub local_experts: u32,
    pub distribution: String,
    pub execution_mode: String,
}

impl AscendMoeChainKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(self.phase.as_str(), "prefill" | "decode")
            || !matches!(self.local_experts, 8 | 16)
            || !matches!(self.distribution.as_str(), "balanced" | "single_expert")
            || !matches!(self.execution_mode.as_str(), "eager" | "graph")
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend MoE chain identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn physical_rows(&self, received: u32) -> u32 {
        if self.phase == "prefill" {
            received
        } else {
            (256 / self.local_experts) * 32 * 8
        }
    }
}

struct Curve {
    values: AxisCurve,
    minimum: u32,
    maximum: u32,
}
type Grids = BTreeMap<AscendMoeChainKey, Curve>;

pub struct AscendMoeChainTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendMoeChainTable {
    pub fn estimate(
        &self,
        key: &AscendMoeChainKey,
        received: u32,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<super::ascend_shape::ShapeEstimate, AicError> {
        match self.query(key, received) {
            Ok(value) => return Ok(super::ascend_shape::ShapeEstimate::measured(value)),
            Err(error)
                if policy == super::ascend_shape::AscendShapePolicy::Anchored
                    && error.is_missing_perf_data() => {}
            Err(error) => return Err(error),
        }
        let curve = self.curve(key)?;
        super::ascend_shape::anchored_axis(
            received,
            curve.minimum,
            curve.maximum,
            &|value| self.query(key, value),
            &|count| {
                let experts = if key.distribution == "balanced" {
                    count.min(f64::from(key.local_experts))
                } else {
                    1.0
                };
                super::ascend_shape::roofline_ms(
                    spec,
                    6.0 * count * 6144.0 * 2048.0,
                    experts * 3.0 * 6144.0 * 2048.0 + count * 6144.0 * 4.0,
                    true,
                )
            },
        )
    }
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
                "Ascend MoE chain rejects filtered primary sources".into(),
            ));
        }
        let path = primary.map(|source| source.source.0);
        if let Some(path) = &path {
            let family = data_root
                .parent()
                .and_then(Path::parent)
                .and_then(|system| {
                    Some(
                        system
                            .join("moe")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend MoE chain requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    fn curve(&self, key: &AscendMoeChainKey) -> Result<&Curve, AicError> {
        key.validate()?;
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = self.path.as_ref().filter(|path| path.is_file()) else {
                    return Err((false, "Ascend MoE chain primary data is missing".into()));
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
        grids.get(key).ok_or_else(|| {
            AicError::PerfDatabase(format!("unmeasured Ascend MoE chain tuple: {key:?}"))
        })
    }

    pub fn ensure_key(&self, key: &AscendMoeChainKey) -> Result<(), AicError> {
        self.curve(key).map(|_| ())
    }

    pub fn query(&self, key: &AscendMoeChainKey, received: u32) -> Result<f64, AicError> {
        let curve = self.curve(key)?;
        if received < curve.minimum || received > curve.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend MoE received tokens={received} outside measured [{}, {}]",
                curve.minimum, curve.maximum
            )));
        }
        curve.values.query(f64::from(received), &|_| 1.0)
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let phase = reader.col("phase")?;
    let experts = reader.col("local_experts")?;
    let distribution = reader.col("distribution")?;
    let mode = reader.col("execution_mode")?;
    let received = reader.col("received_tokens")?;
    let physical = reader.col("physical_rows")?;
    let counts = reader.col("expert_counts")?;
    let hidden = reader.col("hidden_size")?;
    let intermediate = reader.col("intermediate_size")?;
    let group_type = reader.col("group_list_type")?;
    let format = reader.col("weight_format")?;
    let input_dtype = reader.col("input_dtype")?;
    let output_dtype = reader.col("output_dtype")?;
    let gmm = reader.col("gmm_source")?;
    let activation = reader.col("activation_source")?;
    let latency = reader.col("latency_ms")?;
    let cv = reader.col("cv")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let verification = reader.col("execution_verification")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut raw: BTreeMap<AscendMoeChainKey, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut identity: Option<String> = None;
    for row in reader.rows()? {
        let row = row?;
        let key = AscendMoeChainKey {
            phase: row.str_owned(phase)?,
            local_experts: row.u32(experts)?,
            distribution: row.str_owned(distribution)?,
            execution_mode: row.str_owned(mode)?,
        };
        key.validate()?;
        let count = row.u32(received)?;
        let group: Vec<u32> = serde_json::from_str(row.str(counts)?)
            .map_err(|error| AicError::InvalidPerfData(format!("invalid MoE counts: {error}")))?;
        let counts_valid = group.len() == key.local_experts as usize
            && group.iter().enumerate().all(|(index, value)| {
                *value
                    == if key.distribution == "balanced" {
                        count / key.local_experts
                            + u32::from((index as u32) < count % key.local_experts)
                    } else if index == 0 {
                        count
                    } else {
                        0
                    }
            });
        let measured = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let expected = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if !counts_valid
            || count == 0
            || row.u32(physical)? != key.physical_rows(count)
            || count > row.u32(physical)?
            || row.u32(hidden)? != 6144
            || row.u32(intermediate)? != 2048
            || row.u32(group_type)? != 1
            || row.u32(format)? != 29
            || row.str(input_dtype)? != "int8"
            || row.str(output_dtype)? != "bfloat16"
            || row.str(gmm)? != "sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul"
            || row.str(activation)? != "sgl_kernel_npu.activation.swiglu_quant.swiglu_quant"
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(verification)? != expected
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend MoE chain row: {key:?}"
            )));
        }
        if identity
            .as_ref()
            .is_some_and(|expected| expected != runtime)
        {
            return Err(AicError::InvalidPerfData(
                "Ascend MoE chain mixes runtime identities".into(),
            ));
        }
        if identity.is_none() {
            identity = Some(runtime.to_owned());
        }
        if raw
            .entry(key)
            .or_default()
            .insert(count, measured)
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend MoE chain physical key".into(),
            ));
        }
    }
    if raw.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend MoE chain table".into(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(key, points)| {
            (
                key,
                Curve {
                    minimum: *points.first_key_value().unwrap().0,
                    maximum: *points.last_key_value().unwrap().0,
                    values: AxisCurve::from_map("received_tokens", points),
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::ascend_moe_chain::AscendMoeChainOp;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perf_database::{PerfDbSources, PerfSource};

    fn key() -> AscendMoeChainKey {
        AscendMoeChainKey {
            phase: "decode".into(),
            local_experts: 16,
            distribution: "balanced".into(),
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("phase", vec!["decode"; 2]),
            Col::I64("local_experts", vec![16; 2]),
            Col::Str("distribution", vec!["balanced"; 2]),
            Col::Str("execution_mode", vec!["graph"; 2]),
            Col::I64("received_tokens", vec![16, 32]),
            Col::I64("physical_rows", vec![4096; 2]),
            Col::Str(
                "expert_counts",
                vec![
                    "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
                    "[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]",
                ],
            ),
            Col::I64("hidden_size", vec![6144; 2]),
            Col::I64("intermediate_size", vec![2048; 2]),
            Col::I64("group_list_type", vec![1; 2]),
            Col::I64("weight_format", vec![29; 2]),
            Col::Str("input_dtype", vec!["int8"; 2]),
            Col::Str("output_dtype", vec!["bfloat16"; 2]),
            Col::Str(
                "gmm_source",
                vec!["sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul"; 2],
            ),
            Col::Str(
                "activation_source",
                vec!["sgl_kernel_npu.activation.swiglu_quant.swiglu_quant"; 2],
            ),
            Col::F64("latency_ms", vec![1.0, 5.0]),
            Col::F64("cv", vec![0.01; 2]),
            Col::I64("numerical_checks", vec![1024; 2]),
            Col::I64("sample_count", vec![7; 2]),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 2]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 2],
            ),
        ]
    }

    #[test]
    fn received_counts_interpolate_without_treating_capacity_as_work() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendMoeChainTable::new(Some(path));
        assert_eq!(table.query(&key(), 16).unwrap(), 1.0);
        assert_eq!(table.query(&key(), 24).unwrap(), 3.0);
        assert!(
            table
                .query(&key(), 4096)
                .unwrap_err()
                .is_missing_perf_data()
        );
        let mut other = key();
        other.distribution = "single_expert".into();
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
        other = key();
        other.local_experts = 8;
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
        other = key();
        other.phase = "prefill".into();
        assert!(table.query(&other, 24).unwrap_err().is_missing_perf_data());
    }

    #[test]
    fn count_vectors_precision_and_execution_must_match_the_contract() {
        for (index, replacement) in [
            (5, Col::I64("physical_rows", vec![16, 32])),
            (
                6,
                Col::Str(
                    "expert_counts",
                    vec!["[16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]"; 2],
                ),
            ),
            (9, Col::I64("group_list_type", vec![0; 2])),
            (10, Col::I64("weight_format", vec![2; 2])),
            (11, Col::Str("input_dtype", vec!["bfloat16"; 2])),
            (16, Col::F64("cv", vec![0.2; 2])),
            (
                19,
                Col::Str("execution_verification", vec!["eager_invocation"; 2]),
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            assert!(matches!(
                AscendMoeChainTable::new(Some(path)).ensure_key(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        assert!(
            AscendMoeChainTable::new(None)
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn input_padding_precedes_routing_ratio_and_scales_are_resident() {
        let mut operation = AscendMoeChainOp {
            name: "experts".into(),
            scale_factor: 1.0,
            key: key(),
            token_numerator: 8,
            token_denominator: 16,
            input_token_multiple: 8,
        };
        assert_eq!(operation.received_tokens(9).unwrap(), 8);
        assert_eq!(operation.received_tokens(0).unwrap(), 0);
        assert_eq!(operation.weight_bytes(), 604_962_816.0);
        operation.token_numerator = u32::MAX;
        operation.token_denominator = 1;
        assert!(operation.received_tokens(9).is_err());
    }

    #[test]
    fn primary_source_is_exact_and_does_not_override_veto() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/ascend910c/moe/sglang/0.5.22",
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
                AscendMoeChainTable::with_sources(&requested, &resolver)
                    .unwrap()
                    .query(&key(), 24)
                    .unwrap(),
                3.0
            );
        }
        let resolver = SourceResolver::fixed(PerfDbSources::from([(
            BASENAME.into(),
            vec![PerfSource(
                root.path()
                    .join("data/other/moe/sglang/0.5.22")
                    .join(BASENAME),
                None,
            )],
        )]));
        assert!(matches!(
            AscendMoeChainTable::with_sources(&requested, &resolver),
            Err(AicError::InvalidPerfData(_))
        ));
        let resolver = SourceResolver::fixed(PerfDbSources::from([(BASENAME.into(), Vec::new())]));
        assert!(
            AscendMoeChainTable::with_sources(&requested, &resolver)
                .unwrap()
                .ensure_key(&key())
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
