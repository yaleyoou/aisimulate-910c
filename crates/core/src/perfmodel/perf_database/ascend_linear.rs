use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::LeafAxisCurve;
use super::parquet_loader::PerfReader;
use super::perf_interp::LeafValue;
use crate::common::error::AicError;

pub const BASENAME: &str = "ascend_linear_perf.parquet";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendLinearKey {
    pub quant_mode: String,
    pub n: u32,
    pub k: u32,
    pub weight_format: u32,
    pub execution_mode: String,
}

impl AscendLinearKey {
    pub fn validate(&self) -> Result<(), AicError> {
        if !matches!(
            self.quant_mode.as_str(),
            "w8a8" | "w8a8_dynamic" | "bfloat16" | "float32_bfloat16"
        ) || !matches!(self.execution_mode.as_str(), "eager" | "graph")
            || !matches!(self.weight_format, 2 | 29)
            || self.n == 0
            || self.k == 0
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid Ascend linear identity: {self:?}"
            )));
        }
        Ok(())
    }

    pub fn kernel_source(&self) -> &'static str {
        match self.quant_mode.as_str() {
            "w8a8" => {
                "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8LinearMethod"
            }
            "w8a8_dynamic" => {
                "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8DynamicLinearMethod"
            }
            _ => "torch.nn.functional.linear",
        }
    }

    pub fn boundary_dtype(&self) -> &'static str {
        if self.quant_mode == "float32_bfloat16" {
            "float32"
        } else {
            "bfloat16"
        }
    }
}

struct Curve {
    points: LeafAxisCurve<u32>,
    minimum: u32,
    maximum: u32,
}

impl Curve {
    fn query(&self, tokens: u32) -> Result<LeafValue, AicError> {
        if tokens < self.minimum || tokens > self.maximum {
            return Err(AicError::PerfDatabase(format!(
                "Ascend linear M={tokens} is outside measured [{}, {}]",
                self.minimum, self.maximum
            )));
        }
        self.points.query(f64::from(tokens), &|_| 1.0)
    }
}

type Grids = BTreeMap<AscendLinearKey, [Option<Curve>; 2]>;

pub struct AscendLinearTable {
    path: Option<PathBuf>,
    grids: OnceLock<Result<Grids, (bool, String)>>,
}

impl AscendLinearTable {
    pub fn estimate(
        &self,
        key: &AscendLinearKey,
        quant_bias: bool,
        tokens: u32,
        policy: super::ascend_shape::AscendShapePolicy,
        spec: &crate::common::system_spec::SystemSpec,
    ) -> Result<super::ascend_shape::ShapeEstimate, AicError> {
        match self.query(key, quant_bias, tokens) {
            Ok(value) => return Ok(super::ascend_shape::ShapeEstimate::measured(value.latency)),
            Err(error)
                if policy == super::ascend_shape::AscendShapePolicy::Anchored
                    && error.is_missing_perf_data() => {}
            Err(error) => return Err(error),
        }
        let curve = self.curve(key, quant_bias)?;
        super::ascend_shape::anchored_axis(
            tokens,
            curve.minimum,
            curve.maximum,
            &|value| {
                self.query(key, quant_bias, value)
                    .map(|result| result.latency)
            },
            &|count| {
                super::ascend_shape::linear_work(
                    spec,
                    count,
                    f64::from(key.n),
                    f64::from(key.k),
                    key.quant_mode.starts_with("w8a8"),
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
                "Ascend linear does not accept filtered primary kernel sources".into(),
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
                            .join("gemm")
                            .join(data_root.parent()?.file_name()?)
                            .join(data_root.file_name()?)
                            .join(BASENAME),
                    )
                });
            if path != &data_root.join(BASENAME) && family_path.as_ref() != Some(path) {
                return Err(AicError::InvalidPerfData(
                    "Ascend linear requires the exact requested system/backend/version".into(),
                ));
            }
        }
        Ok(Self::new(path))
    }

    pub fn ensure_key(&self, key: &AscendLinearKey, quant_bias: bool) -> Result<(), AicError> {
        self.curve(key, quant_bias).map(|_| ())
    }

    pub fn query(
        &self,
        key: &AscendLinearKey,
        quant_bias: bool,
        tokens: u32,
    ) -> Result<LeafValue, AicError> {
        self.curve(key, quant_bias)?.query(tokens)
    }

    fn curve(&self, key: &AscendLinearKey, quant_bias: bool) -> Result<&Curve, AicError> {
        key.validate()?;
        if quant_bias && key.quant_mode != "w8a8" {
            return Err(AicError::InvalidEngineConfig(
                "quant_bias is specific to static W8A8".into(),
            ));
        }
        let grids = self
            .grids
            .get_or_init(|| {
                let Some(path) = &self.path else {
                    return Err((false, "Ascend linear primary data is missing".into()));
                };
                if !path.is_file() {
                    return Err((
                        false,
                        format!("Ascend linear data is missing: {}", path.display()),
                    ));
                }
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
            .and_then(|curves| curves[usize::from(quant_bias)].as_ref())
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "Ascend linear tuple is unmeasured: {key:?}, quant_bias={quant_bias}"
                ))
            })
    }
}

fn load(path: &Path) -> Result<Grids, AicError> {
    let reader = PerfReader::open(path)?;
    let quant = reader.col("quant_mode")?;
    let output_width = reader.col("n")?;
    let input_width = reader.col("k")?;
    let format = reader.col("weight_format")?;
    let execution = reader.col("execution_mode")?;
    let bias = reader.col("quant_bias")?;
    let tokens = reader.col("m")?;
    let latency = reader.col("latency_ms")?;
    let input_dtype = reader.col("input_dtype")?;
    let output_dtype = reader.col("output_dtype")?;
    let kernel = reader.col("kernel_source")?;
    let verification = reader.col("execution_verification")?;
    let checks = reader.col("numerical_checks")?;
    let samples = reader.col("sample_count")?;
    let cv = reader.col("cv")?;
    let provenance = reader.col("runtime_identity_sha256")?;
    let mut identities: BTreeMap<(AscendLinearKey, bool), String> = BTreeMap::new();
    let mut rows: BTreeMap<AscendLinearKey, [BTreeMap<u32, LeafValue>; 2]> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = AscendLinearKey {
            quant_mode: row.str_owned(quant)?,
            n: row.u32(output_width)?,
            k: row.u32(input_width)?,
            weight_format: row.u32(format)?,
            execution_mode: row.str_owned(execution)?,
        };
        key.validate()?;
        let has_bias = row.bool_strict(bias)?;
        let count = row.u32(tokens)?;
        let measured = row.f64(latency)?;
        let variation = row.f64(cv)?;
        let runtime = row.str(provenance)?;
        let verification_mode = if key.execution_mode == "graph" {
            "captured_graph_replay"
        } else {
            "eager_invocation"
        };
        if count == 0
            || !measured.is_finite()
            || measured <= 0.0
            || !variation.is_finite()
            || !(0.0..=0.1).contains(&variation)
            || row.u32(checks)? == 0
            || row.u32(samples)? < 3
            || row.str(input_dtype)? != key.boundary_dtype()
            || row.str(output_dtype)? != key.boundary_dtype()
            || row.str(kernel)? != key.kernel_source()
            || row.str(verification)? != verification_mode
            || (has_bias && key.quant_mode != "w8a8")
            || runtime.len() != 64
            || !runtime.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AicError::InvalidPerfData(format!(
                "unqualified Ascend linear row: {key:?}"
            )));
        }
        let expected = identities
            .entry((key.clone(), has_bias))
            .or_insert_with(|| runtime.to_owned());
        if expected != runtime {
            return Err(AicError::InvalidPerfData(
                "Ascend linear curve mixes runtime identities".into(),
            ));
        }
        if rows.entry(key).or_default()[usize::from(has_bias)]
            .insert(count, LeafValue::latency_only(measured))
            .is_some()
        {
            return Err(AicError::InvalidPerfData(
                "duplicate Ascend linear physical key".into(),
            ));
        }
    }
    if rows.is_empty() {
        return Err(AicError::InvalidPerfData(
            "empty Ascend linear table".into(),
        ));
    }
    Ok(rows
        .into_iter()
        .map(|(key, curves)| {
            (
                key,
                curves.map(|points| {
                    if points.is_empty() {
                        None
                    } else {
                        Some(Curve {
                            minimum: *points.first_key_value().unwrap().0,
                            maximum: *points.last_key_value().unwrap().0,
                            points: LeafAxisCurve::from_map("m", points),
                        })
                    }
                }),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PerfDbSources, PerfSource};
    use crate::operators::ascend_linear::{AscendLinearBias, AscendLinearOp, AscendLinearTokens};
    use crate::perf_database::PerfDatabase;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> AscendLinearKey {
        AscendLinearKey {
            quant_mode: "w8a8".into(),
            n: 128,
            k: 64,
            weight_format: 29,
            execution_mode: "graph".into(),
        }
    }

    fn columns() -> Vec<Col> {
        vec![
            Col::Str("quant_mode", vec!["w8a8"; 4]),
            Col::I64("n", vec![128; 4]),
            Col::I64("k", vec![64; 4]),
            Col::I64("weight_format", vec![29; 4]),
            Col::Str("execution_mode", vec!["graph"; 4]),
            Col::Bool("quant_bias", vec![false, false, true, true]),
            Col::I64("m", vec![1, 9, 1, 9]),
            Col::F64("latency_ms", vec![1.0, 5.0, 2.0, 4.0]),
            Col::Str("input_dtype", vec!["bfloat16"; 4]),
            Col::Str("output_dtype", vec!["bfloat16"; 4]),
            Col::Str(
                "kernel_source",
                vec![
                    "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8LinearMethod";
                    4
                ],
            ),
            Col::Str("execution_verification", vec!["captured_graph_replay"; 4]),
            Col::I64("numerical_checks", vec![96; 4]),
            Col::I64("sample_count", vec![9; 4]),
            Col::F64("cv", vec![0.01; 4]),
            Col::Str(
                "runtime_identity_sha256",
                vec!["1111111111111111111111111111111111111111111111111111111111111111"; 4],
            ),
        ]
    }

    fn operation() -> AscendLinearOp {
        AscendLinearOp {
            name: "projection".into(),
            scale_factor: 2.0,
            key: key(),
            bias: AscendLinearBias::TpGroupMax,
            token_axis: AscendLinearTokens::Tokens,
            token_split: 1,
            token_multiple: 1,
        }
    }

    #[test]
    fn measured_brackets_keep_bias_layout_and_execution_exact() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        write_parquet(&path, &columns());
        let table = AscendLinearTable::new(Some(path));
        assert_eq!(table.query(&key(), false, 1).unwrap().latency, 1.0);
        assert_eq!(table.query(&key(), true, 1).unwrap().latency, 2.0);
        assert_eq!(table.query(&key(), false, 5).unwrap().latency, 3.0);
        assert_eq!(table.query(&key(), true, 5).unwrap().latency, 3.0);
        assert!(
            table
                .query(&key(), false, 10)
                .unwrap_err()
                .is_missing_perf_data()
        );
        let mut changed = key();
        changed.execution_mode = "eager".into();
        assert!(
            table
                .query(&changed, false, 5)
                .unwrap_err()
                .is_missing_perf_data()
        );
        changed = key();
        changed.weight_format = 2;
        assert!(
            table
                .query(&changed, false, 5)
                .unwrap_err()
                .is_missing_perf_data()
        );
        changed = key();
        changed.quant_mode = "w8a8_dynamic".into();
        assert!(matches!(
            table.query(&changed, true, 5),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }

    #[test]
    fn collection_identity_is_separate_per_bias_curve_but_never_mixed_inside_one() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let first = "1111111111111111111111111111111111111111111111111111111111111111";
        let second = "2222222222222222222222222222222222222222222222222222222222222222";
        let mut fixture = columns();
        *fixture.last_mut().unwrap() = Col::Str(
            "runtime_identity_sha256",
            vec![first, first, second, second],
        );
        write_parquet(&path, &fixture);
        let table = AscendLinearTable::new(Some(path.clone()));
        assert_eq!(table.query(&key(), false, 5).unwrap().latency, 3.0);
        assert_eq!(table.query(&key(), true, 5).unwrap().latency, 3.0);
        *fixture.last_mut().unwrap() = Col::Str(
            "runtime_identity_sha256",
            vec![first, second, second, second],
        );
        write_parquet(&path, &fixture);
        assert!(matches!(
            AscendLinearTable::new(Some(path)).query(&key(), false, 5),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn actual_group_max_does_not_assume_bias_is_always_slower() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("data/ascend910c/sglang/0.5.22");
        std::fs::create_dir_all(&directory).unwrap();
        write_parquet(&directory.join(BASENAME), &columns());
        std::fs::write(root.path().join("ascend910c.yaml"),
            "data_dir: data/ascend910c\ngpu:\n  mem_bw: 1\n  int8_tc_flops: 1\nnode:\n  num_gpus_per_node: 16\n  intra_node_bw: 1\n  inter_node_bw: 1\nmisc: {}\n").unwrap();
        let database = PerfDatabase::load(root.path(), "ascend910c", "sglang", "0.5.22").unwrap();
        let op = operation();
        op.ensure_ready(&database).unwrap();
        assert_eq!(op.query(&database, 1, 1).unwrap().latency_ms, 4.0);
        assert_eq!(op.query(&database, 9, 1).unwrap().latency_ms, 10.0);
        assert_eq!(op.weight_bytes(), 20232.0);
        assert_eq!(op.query(&database, 0, 0).unwrap().latency_ms, 0.0);
    }

    #[test]
    fn mixed_indexer_precision_cannot_use_bfloat16_input_rows() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(BASENAME);
        let mut data = columns();
        data[0] = Col::Str("quant_mode", vec!["float32_bfloat16"; 4]);
        data[5] = Col::Bool("quant_bias", vec![false; 4]);
        data[6] = Col::I64("m", vec![1, 9, 17, 25]);
        data[8] = Col::Str("input_dtype", vec!["float32"; 4]);
        data[9] = Col::Str("output_dtype", vec!["float32"; 4]);
        data[10] = Col::Str("kernel_source", vec!["torch.nn.functional.linear"; 4]);
        write_parquet(&path, &data);
        let table = AscendLinearTable::new(Some(path.clone()));
        let mut mixed = key();
        mixed.quant_mode = "float32_bfloat16".into();
        assert_eq!(table.query(&mixed, false, 5).unwrap().latency, 3.0);
        let mut bf16 = mixed.clone();
        bf16.quant_mode = "bfloat16".into();
        assert!(
            table
                .query(&bf16, false, 5)
                .unwrap_err()
                .is_missing_perf_data()
        );
        data[8] = Col::Str("input_dtype", vec!["bfloat16"; 4]);
        write_parquet(&path, &data);
        let table = AscendLinearTable::new(Some(path));
        assert!(matches!(
            table.query(&mixed, false, 5),
            Err(AicError::InvalidPerfData(_))
        ));
    }

    #[test]
    fn token_axis_sharding_and_padding_are_explicit() {
        let mut op = operation();
        op.token_split = 2;
        op.token_multiple = 8;
        assert_eq!(op.measured_tokens(18, 3).unwrap(), 16);
        op.token_axis = AscendLinearTokens::Requests;
        assert_eq!(op.measured_tokens(8192, 3).unwrap(), 8);
        assert_eq!(op.measured_tokens(0, 0).unwrap(), 0);
        op.token_split = 0;
        assert!(op.measured_tokens(8, 8).is_err());
        op.token_split = 1;
        assert!(op.measured_tokens(0, u32::MAX).is_err());
    }

    #[test]
    fn malformed_data_never_becomes_a_coverage_miss() {
        let replacements = [
            (6, Col::I64("m", vec![1; 4])),
            (
                11,
                Col::Str("execution_verification", vec!["eager_invocation"; 4]),
            ),
            (12, Col::I64("numerical_checks", vec![0; 4])),
            (14, Col::F64("cv", vec![0.3; 4])),
        ];
        for (index, replacement) in replacements {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            let mut data = columns();
            data[index] = replacement;
            write_parquet(&path, &data);
            let table = AscendLinearTable::new(Some(path));
            assert!(matches!(
                table.query(&key(), false, 1),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        let table = AscendLinearTable::new(None);
        assert!(
            table
                .query(&key(), false, 1)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn exact_primary_accepts_both_layouts_and_respects_veto() {
        let root = tempfile::tempdir().unwrap();
        let system = root.path().join("data/ascend910c");
        let requested = system.join("sglang/0.5.22");
        for path in [
            requested.join(BASENAME),
            system.join("gemm/sglang/0.5.22").join(BASENAME),
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            write_parquet(&path, &columns());
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(path, None)],
            )]));
            let table = AscendLinearTable::with_sources(&requested, &resolver).unwrap();
            assert_eq!(table.query(&key(), false, 5).unwrap().latency, 3.0);
        }
        let resolver = SourceResolver::fixed(PerfDbSources::from([(BASENAME.into(), Vec::new())]));
        let table = AscendLinearTable::with_sources(&requested, &resolver).unwrap();
        assert!(
            table
                .query(&key(), false, 5)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }

    #[test]
    fn primary_cannot_relabel_another_system_backend_or_version() {
        let root = tempfile::tempdir().unwrap();
        let requested = root.path().join("data/ascend910c/sglang/0.5.22");
        for relative in [
            "data/other/gemm/sglang/0.5.22",
            "data/ascend910c/gemm/vllm/0.5.22",
            "data/ascend910c/gemm/sglang/0.5.21",
        ] {
            let resolver = SourceResolver::fixed(PerfDbSources::from([(
                BASENAME.into(),
                vec![PerfSource(root.path().join(relative).join(BASENAME), None)],
            )]));
            assert!(matches!(
                AscendLinearTable::with_sources(&requested, &resolver),
                Err(AicError::InvalidPerfData(_))
            ));
        }
        let resolver = SourceResolver::live(super::super::ResolveCtx {
            systems_root: root.path().into(),
            system_data_root: root.path().join("data/ascend910c"),
            backend: "sglang".into(),
            version: "0.5.22".into(),
            enable_shared_layer: true,
            strict: false,
        });
        let sibling = root.path().join("data/ascend910c/gemm/sglang/0.5.21");
        std::fs::create_dir_all(&sibling).unwrap();
        write_parquet(&sibling.join(BASENAME), &columns());
        let table = AscendLinearTable::with_sources(&requested, &resolver).unwrap();
        assert!(
            table
                .query(&key(), false, 5)
                .unwrap_err()
                .is_missing_perf_data()
        );
    }
}
