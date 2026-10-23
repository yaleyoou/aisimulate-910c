use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::ascend_linear::AscendLinearTokens;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_vector::AscendVectorKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendVectorOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendVectorKey,
    pub token_axis: AscendLinearTokens,
    pub token_split: u32,
    pub token_multiple: u32,
}

impl AscendVectorOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.token_split == 0
            || self.token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend vector token/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn measured_tokens(&self, tokens: u32, requests: u32) -> Result<u32, AicError> {
        self.validate()?;
        let count = match self.token_axis {
            AscendLinearTokens::Tokens => tokens,
            AscendLinearTokens::Requests => requests,
        };
        count
            .div_ceil(self.token_split)
            .div_ceil(self.token_multiple)
            .checked_mul(self.token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend vector token rounding overflow".into())
            })
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend vector requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend vector requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_vector.ensure_key(&self.key)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        tokens: u32,
        requests: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        let count = self.measured_tokens(tokens, requests)?;
        if count == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        Ok(
            PerformanceResult::new(db.ascend_vector.query(&self.key, count)?, Source::Silicon)
                .scaled(self.scale_factor),
        )
    }

    pub fn weight_bytes(&self) -> f64 {
        self.key.parameter_bytes() * self.scale_factor
    }
}
