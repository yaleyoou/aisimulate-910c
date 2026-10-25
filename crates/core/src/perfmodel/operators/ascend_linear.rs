use serde::{Deserialize, Serialize};

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use crate::perf_database::ascend_linear::AscendLinearKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AscendLinearBias {
    Never,
    Always,
    TpGroupMax,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AscendLinearTokens {
    #[default]
    Tokens,
    Requests,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscendLinearOp {
    pub name: String,
    pub scale_factor: f64,
    pub key: AscendLinearKey,
    pub bias: AscendLinearBias,
    pub token_axis: AscendLinearTokens,
    pub token_split: u32,
    pub token_multiple: u32,
}

impl AscendLinearOp {
    pub fn validate(&self) -> Result<(), AicError> {
        self.key.validate()?;
        if self.token_split == 0
            || self.token_multiple == 0
            || !self.scale_factor.is_finite()
            || self.scale_factor < 0.0
            || (self.bias != AscendLinearBias::Never && self.key.quant_mode != "w8a8")
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid Ascend linear shape/bias policy".into(),
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
                AicError::InvalidEngineConfig("Ascend linear token rounding overflow".into())
            })
    }

    pub fn ensure_ready(&self, db: &PerfDatabase) -> Result<(), AicError> {
        self.validate()?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend linear requires SGLang".into(),
            ));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend linear requires measured data; no validated analytic fallback".into(),
            ));
        }
        if self.scale_factor == 0.0 {
            return Ok(());
        }
        match self.bias {
            AscendLinearBias::Never => db.ascend_linear.ensure_key(&self.key, false),
            AscendLinearBias::Always => db.ascend_linear.ensure_key(&self.key, true),
            AscendLinearBias::TpGroupMax => {
                db.ascend_linear.ensure_key(&self.key, true)?;
                db.ascend_linear.ensure_key(&self.key, false)
            }
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        tokens: u32,
        requests: u32,
    ) -> Result<PerformanceResult, AicError> {
        let count = self.measured_tokens(tokens, requests)?;
        if db.backend != "sglang" {
            return Err(AicError::InvalidEngineConfig(
                "Ascend linear requires SGLang".into(),
            ));
        }
        if count == 0 || self.scale_factor == 0.0 {
            return Ok(PerformanceResult::new(0.0, Source::Empirical));
        }
        if !matches!(
            db.database_mode,
            DatabaseMode::Silicon | DatabaseMode::Hybrid
        ) {
            return Err(AicError::EmpiricalNotImplemented(
                "Ascend linear requires measured data".into(),
            ));
        }
        let estimate = |bias| {
            db.ascend_linear.estimate(
                &self.key,
                bias,
                count,
                db.ascend_shape_policy,
                &db.system_spec,
            )
        };
        let value = match self.bias {
            AscendLinearBias::Never => estimate(false)?,
            AscendLinearBias::Always => estimate(true)?,
            AscendLinearBias::TpGroupMax => {
                let biased = estimate(true)?;
                let plain = estimate(false)?;
                crate::perf_database::ascend_shape::ShapeEstimate {
                    latency_ms: biased.latency_ms.max(plain.latency_ms),
                    extrapolated: biased.extrapolated || plain.extrapolated,
                }
            }
        };
        Ok(value.result().scaled(self.scale_factor))
    }

    pub fn weight_bytes(&self) -> f64 {
        let output = f64::from(self.key.n);
        let input = f64::from(self.key.k);
        let bytes = match self.key.quant_mode.as_str() {
            "bfloat16" | "float32_bfloat16" => 2.0 * output * input,
            "w8a8" => output * input + 12.0 * output + 6.0 * input + 4.0,
            _ => output * input + 4.0 * output,
        };
        bytes * self.scale_factor
    }
}
