# gp_detrend.py
# TinyGP + NumPyro로 라이트커브 GP 디트렌드 (조건부 평균을 predict()로만 계산)
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import tinygp
from tinygp import kernels


# ---------- 유틸 ----------
def _to_days(t, unit="days"):
    factors = dict(days=1.0, hours=1/24.0, minutes=1/(24*60.0), seconds=1/(24*3600.0))
    if unit not in factors:
        raise ValueError(f"Unsupported unit: {unit}")
    return np.asarray(t, dtype=float) * factors[unit]


# ---------- GP 구성 ----------
def _build_gp(t, yerr, log_sigma, log_rho, mu=None, log_sigma_n=None):
    sigma = jnp.exp(log_sigma)
    rho   = jnp.exp(log_rho)

    # ⬇︎ tinygp 커널: 여러분 파일의 스타일에 맞추세요.
    # (a) 기존에 'kernels.Matern32(scale=...)' 를 쓰고 있었다면:
    k = (sigma**2) * kernels.Matern32(scale=rho)
    # (b) 만약 'kernels.stationary.Matern32(scale=...)' 를 쓰고 있었다면 위 한 줄을 아래로 바꾸세요.
    # k = (sigma**2) * kernels.stationary.Matern32(scale=rho)

    mean = 0.0 if mu is None else mu

    # 관측 잡음(diagonal)
    diag_meas = (yerr**2) if (yerr is not None) else 0.0
    # 학습되는 균일 화이트 노이즈 항
    diag_white = (jnp.exp(log_sigma_n)**2) if (log_sigma_n is not None) else 0.0

    diag = diag_meas + diag_white + 1e-12  # 숫자 안정화용 jitter

    return tinygp.GaussianProcess(k, t, diag=diag, mean=mean)


# ---------- NumPyro 모델 ----------
def _model(t, y=None, yerr=None, use_const_mean=True,
           transit_duration_hours=2.0, rho_mult=5.0):

    # 지속시간(일)
    dur_days = transit_duration_hours / 24.0

    # rho(커널 스케일)의 탐색 범위: 최소는 (rho_mult × 지속시간), 최대는 베이스라인 길이
    rho_min = jnp.maximum(rho_mult * dur_days, 1e-4)
    rho_max = jnp.maximum(t.max() - t.min(), rho_min + 1e-3)

    # 커널 스케일 & 스케일 팩터
    log_sigma = numpyro.sample("log_sigma", dist.Uniform(jnp.log(1e-6), jnp.log(1.0)))
    log_rho   = numpyro.sample("log_rho",   dist.Uniform(jnp.log(rho_min), jnp.log(rho_max)))

    # 🔑 균일 화이트 노이즈(관측 노이즈) 학습
    log_sigma_n = numpyro.sample("log_sigma_n", dist.Uniform(jnp.log(1e-7), jnp.log(0.5)))

    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0)) if use_const_mean else None

    gp = _build_gp(t, yerr, log_sigma, log_rho, mu=mu, log_sigma_n=log_sigma_n)
    numpyro.factor("gp_loglike", gp.log_probability(y))

# ---------- 사후에서 predict만 사용 ----------
def _posterior_noise_mean_via_predict(t, y, yerr, samples, use_const_mean=True):
    def one_predict(ls, lr, mu, lsn):
        gp = _build_gp(t, yerr, ls, lr, (mu if use_const_mean else None), log_sigma_n=lsn)
        return gp.predict(y, t)

    log_sigma_s   = samples["log_sigma"]
    log_rho_s     = samples["log_rho"]
    log_sigma_n_s = samples["log_sigma_n"]
    mu_s = samples["mu"] if use_const_mean else jnp.zeros_like(log_sigma_s)

    mean_stack = vmap(one_predict)(log_sigma_s, log_rho_s, mu_s, log_sigma_n_s)
    return jnp.mean(mean_stack, axis=0)


def _run_mcmc(t, y, yerr, use_const_mean, num_warmup, num_samples, chains, seed,
                  transit_duration_hours=2.0, rho_mult=5.0):
    nuts = NUTS(_model, target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=chains, progress_bar=True)
    mcmc.run(
        jax.random.key(seed),
        t=t, y=y, yerr=yerr, use_const_mean=use_const_mean,
        transit_duration_hours=transit_duration_hours,
        rho_mult=rho_mult                               
    )
    mcmc.print_summary()
    return mcmc


# ---------- 공개 API ----------
def detrend_df(df, time="time", flux="flux", err="error",
               unit="days", center_flux=True, mean_const=True,
               samples=800, warmup=800, chains=2, seed=42,
               transit_duration_hours=2.0, rho_mult=5.0):

    t_days = _to_days(df[time], unit)
    t_cent = t_days - t_days.min()
    y = df[flux].to_numpy(float)

    # error 컬럼이 없거나 NaN이 섞여 있으면 None으로 두어
    # 학습되는 화이트노이즈가 대신 책임지게 합니다.
    if (err in df.columns) and df[err].notna().all():
        yerr = df[err].to_numpy(float)
    else:
        yerr = None

    shift = np.median(y) if center_flux else 0.0
    y = y - shift

    mcmc = _run_mcmc(
        t=jnp.array(t_cent),
        y=jnp.array(y),
        yerr=(jnp.array(yerr) if yerr is not None else None),
        use_const_mean=mean_const,
        num_warmup=warmup,
        num_samples=samples,
        chains=chains,
        seed=seed,
        transit_duration_hours=transit_duration_hours,
        rho_mult=rho_mult
    )
    samples_dict = mcmc.get_samples(group_by_chain=False)

    gp_mean = np.asarray(_posterior_noise_mean_via_predict(
        jnp.array(t_cent), jnp.array(y),
        (jnp.array(yerr) if yerr is not None else None),
        samples_dict, use_const_mean=mean_const
    ))
    ycorr = gp_mean + shift  # 저주파 gp값만 사용하기. 고주파인 노이즈는 빼고 본다.

    out = df.copy()
    out["flux_corrected"] = ycorr
    return out, ycorr, mcmc