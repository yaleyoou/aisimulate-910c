use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_moe_chain::AscendMoeChainKey;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendMoeChainOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendMoeChainKey,
    pub token_numerator: u32,
    pub token_denominator: u32,
    pub input_token_multiple: u32,
}

impl AscendMoeChainOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.token_numerator == 0
            || self.token_denominator == 0
            || self.input_token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend MoE token/scale policy".into(),
            ));
        }
        Ok(())
    }

    pub fn received_tokens(&self, tokens: u32) -> Result<u32, AicError> {
        self.validate()?;
        let padded = tokens
            .div_ceil(self.input_token_multiple)
            .checked_mul(self.input_token_multiple)
            .ok_or_else(|| {
                AicError::InvalidEngineConfig("Ascend MoE token rounding overflow".into())
            })?;
        let routed = u64::from(padded) * u64::from(self.token_numerator);
        u32::try_from(routed.div_ceil(u64::from(self.token_denominator)))
            .map_err(|_| AicError::InvalidEngineConfig("Ascend MoE receive-count overflow".into()))
    }

    fn validate_database(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend MoE chain requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend MoE chain requires measured data; no analytic fallback".into(),
            ));
        }
        Ok(())
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate_database(db)?;
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        db.ascend_moe_chain.ensure_key(&self.key)
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        self.validate_database(db)?;
        let received = self.received_tokens(tokens)?;
        if received == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        Ok(db
            .ascend_moe_chain
            .estimate(&self.key, received, db.ascend_shape_policy, &db.system_spec)?
            .result()
            .scaled(self.scale_factor))
    }

    pub fn weight_bytes(&self) -> f64 {
        f64::from(self.key.local_experts)
            * (6144.0 * 2048.0 * 3.0 + (4096.0 + 6144.0) * 6.0)
            * self.scale_factor
    }
}
