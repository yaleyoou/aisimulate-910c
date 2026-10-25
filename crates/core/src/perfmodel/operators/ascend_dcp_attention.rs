use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_dcp_attention::AscendDcpAttentionKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendDcpAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendDcpAttentionKey,
    pub graph_buckets: Vec<u32>,
}

impl AscendDcpAttentionOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || (self.key.phase == "decode") == self.graph_buckets.is_empty()
            || self.graph_buckets.iter().any(|value| *value == 0)
            || self.graph_buckets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid DCP graph or layer-count policy".into(),
            ));
        }
        Ok(())
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.system != "ascend910c" || db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend DCP requires SGLang on910C".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend DCP requires measured data".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_dcp_attention.ensure_key(&self.key)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch: u32,
        sequence: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        if batch == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        let padded = if self.key.phase == "prefill" {
            batch
        } else {
            self.graph_buckets
                .iter()
                .copied()
                .find(|bucket| *bucket >= batch)
                .ok_or_else(|| {
                    AicError::PerfDatabase("DCP batch exceeds measured graph buckets".into())
                })?
        };
        let past = if self.key.phase == "prefill" {
            prefix
        } else {
            0
        };
        Ok(db
            .ascend_dcp_attention
            .estimate(
                &self.key,
                batch,
                padded,
                sequence,
                past,
                db.ascend_shape_policy,
                &db.system_spec,
            )?
            .result()
            .scaled(self.scale_factor))
    }
}
