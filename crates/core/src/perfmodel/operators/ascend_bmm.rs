use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_bmm::AscendBmmKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendBmmOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendBmmKey,
    pub token_multiple: u32,
    #[serde(default)]
    pub resident_copies: Option<u32>,
}

impl AscendBmmOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.token_multiple == 0 || !self.scale_factor.is_finite() || self.scale_factor < 0.0 {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend BMM token/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn measured_tokens(&self, tokens: u32) -> Result<u32, AicError> {
        self.validate()?;
        tokens
            .div_ceil(self.token_multiple)
            .checked_mul(self.token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend BMM token rounding overflow".into())
            })
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend BMM requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend BMM requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_bmm.ensure_key(&self.key)
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        if tokens == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        Ok(db
            .ascend_bmm
            .estimate(
                &self.key,
                self.measured_tokens(tokens)?,
                db.ascend_shape_policy,
                &db.system_spec,
            )?
            .result()
            .scaled(self.scale_factor))
    }

    pub fn weight_bytes(&self) -> f64 {
        let (inner, width, _, _) = self.key.shape();
        f64::from(self.key.local_heads)
            * f64::from(inner)
            * f64::from(width)
            * 2.0
            * self
                .resident_copies
                .map(f64::from)
                .unwrap_or(self.scale_factor)
    }
}
