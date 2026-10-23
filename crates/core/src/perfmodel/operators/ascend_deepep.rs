use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_deepep::AscendDeepEpKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendDeepEpOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendDeepEpKey,
    pub token_split: u32,
    pub input_token_multiple: u32,
}

impl AscendDeepEpOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.token_split == 0
            || self.input_token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend DeepEP token/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn sent_tokens(&self, tokens: u32) -> Result<u32, AicError> {
        self.validate()?;
        let padded = tokens
            .div_ceil(self.input_token_multiple)
            .checked_mul(self.input_token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend DeepEP token rounding overflow".into())
            })?;
        Ok(padded.div_ceil(self.token_split))
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend DeepEP requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend DeepEP requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_deepep.ensure_key(&self.key)
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        let sent = self.sent_tokens(tokens)?;
        if sent == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        Ok(
            PerformanceResult::new(db.ascend_deepep.query(&self.key, sent)?, Source::Silicon)
                .scaled(self.scale_factor),
        )
    }
}
