# gdsp_delay_linear_wetdry
```python
# gdelay_wetdry.py
"""
MODULE NAME:
gdelay_wetdry

DESCRIPTION:
A fully differentiable, time-varying mono delay line with optional wet/dry mixing.
Supports two modes: integer-sample delay (no interpolation) and linear-interpolated fractional delay (with some spectral coloration).
Implements parameter smoothing for delay time and wet mix, and uses a circular buffer with power-of-two length.

INPUTS:

* x : input audio sample or signal (scalar per-tick, 1D array per-process)
* delay_samples : target delay in samples (float, per-tick or per-sample over time)
* mix : wet mix in [0, 1]; 0 = dry only, 1 = wet only (per-tick or per-sample)
* mode : 0 => integer delay only, 1 => linear interpolation (fractional delay)
* delay_smoothing_coeff : smoothing coefficient in (0, 1]; larger = faster tracking
* mix_smoothing_coeff : smoothing coefficient in (0, 1]; larger = faster tracking

OUTPUTS:

* y : output audio sample or signal (same shape as x)
* new_state : updated delay state tuple

STATE VARIABLES:
(state_buffer, write_idx, delay_smooth, mix_smooth)

where:

* state_buffer : 1D array of length L (power-of-two) holding previous samples
* write_idx    : scalar int index of next write position in the buffer
* delay_smooth : smoothed delay (in samples)
* mix_smooth   : smoothed wet mix

EQUATIONS / MATH:

Let:

* x[n]          : input sample at time n
* y_d[n]        : delayed sample at time n
* y[n]          : output sample at time n
* d_t[n]        : target delay (samples) at time n
* m_t[n]        : target wet mix in [0, 1] at time n
* d_s[n]        : smoothed delay
* m_s[n]        : smoothed mix
* a_d           : delay_smoothing_coeff in (0, 1]
* a_m           : mix_smoothing_coeff in (0, 1]
* L             : buffer length
* w[n]          : write index at time n (position in buffer where x[n] is written *after* reading)
* b[k]          : sample stored at buffer index k

Smoothing:

* d_s[n+1] = d_s[n] + a_d * (d_t[n] - d_s[n])
* m_s[n+1] = m_s[n] + a_m * (m_t[n] - m_s[n])

Clamp:

* d_s_clamped[n+1] = clip(d_s[n+1], 0, d_max)   (d_max = max_delay_samples)
* m_s_clamped[n+1] = clip(m_s[n+1], 0, 1)

Delay indexing (time-varying, circular buffer):

Let:

* d = d_s_clamped[n+1] (float delay in samples)

Integer-only mode (mode == 0):

* d_int = round(d)
* frac  = 0

Linear interpolation mode (mode == 1):

* d_floor = floor(d)
* frac    = d - d_floor
* d_int   = d_floor

Use conditional selection over mode to pick (d_int, frac) without Python branching.

Define floating read position using current write index w[n]:

* r_pos = w[n] - d_int   (interpreted in samples)
* r_pos_wrapped = mod(r_pos, L)

Compute integer indices:

* i0 = floor(r_pos_wrapped)
* i1 = mod(i0 + 1, L)

Read samples from buffer (circular):

* s0 = b[i0]
* s1 = b[i1]

Linear interpolation:

* y_d[n] = (1 - frac) * s0 + frac * s1

Wet/dry mix:

* m = m_s_clamped[n+1]
* dry = 1 - m
* y[n] = dry * x[n] + m * y_d[n]

Write new sample and advance write index:

* b'[w[n]] = x[n]
* w[n+1] = mod(w[n] + 1, L)

State update:

* state[n]   = (b, w[n], d_s[n], m_s[n])
* state[n+1] = (b', w[n+1], d_s[n+1], m_s[n+1])

through-zero rules:

* Delay time is clamped to [0, max_delay_samples].
* Effective minimum delay is one sample due to read-then-write ordering, even when delay_smooth is 0.

phase wrapping rules:

* All buffer indices are wrapped with modulo L using mod(·, L).

nonlinearities:

* None beyond clipping of delay and mix.

interpolation rules:

* mode == 0: integer delay, frac = 0 (no interpolation; all-pass in magnitude, phase-only shift).
* mode == 1: linear interpolation between adjacent samples with weight frac in [0, 1].

time-varying coefficient rules:

* Smoothing is a first-order low-pass in discrete time for both delay and mix:
  d_s[n+1] = (1 - a_d) * d_s[n] + a_d * d_t[n]
  m_s[n+1] = (1 - a_m) * m_s[n] + a_m * m_t[n]

NOTES:

* Recommended parameter ranges:

  * delay_smoothing_coeff, mix_smoothing_coeff in (0, 1]; small values give slower, smoother parameter changes.
  * mix in [0, 1] for stable wet/dry mixing.
  * mode should be 0 or 1; non-integer values will be interpreted via a differentiable conditional, but behavior is only meaningful near 0 or 1.
* The buffer length is a power of two >= max_delay_samples, but the code never relies on power-of-two bit tricks; it uses jnp.mod for index wrapping.
* All operations are written to be JAX-jittable and differentiable with respect to x, delay_samples, and mix (fractional delay path yields gradients w.r.t. delay).

---

A fully differentiable JAX mono delay line with optional wet/dry mixing.

Features:
- Variable delay time in samples (time-varying, differentiable).
- Two modes:
    mode = 0 : integer delay only (no interpolation, phase-only shift).
    mode = 1 : linear interpolation for fractional delay (simple, some coloration).
- Parameter smoothing for delay time and wet mix.
- Pure functional API with tuple-based state and params.
- Designed to be JIT-friendly and differentiable in GDSP style.

Public functions:
- delay_wetdry_init(...)
- delay_wetdry_update_state(...)
- delay_wetdry_tick(x, delay_target, mix_target, state, params)
- delay_wetdry_process(x, delay_target, mix_target, state, params)

State = (buffer, write_idx, delay_smooth, mix_smooth)
Params = (sample_rate, max_delay_samples, buffer_len, mode,
          delay_smoothing_coeff, mix_smoothing_coeff)
"""

import math

import jax
import jax.numpy as jnp
from jax import lax


# -----------------------------------------------------------------------------
# 1. Initialization
# -----------------------------------------------------------------------------


def delay_wetdry_init(
    max_delay_samples,
    sample_rate,
    delay_init_samples,
    mix_init,
    mode,
    delay_smoothing_coeff,
    mix_smoothing_coeff,
    dtype=jnp.float32,
):
    """
    Initialize delay line state and params.

    Args:
        max_delay_samples: maximum delay in samples (Python int or scalar).
        sample_rate: sample rate in Hz (scalar).
        delay_init_samples: initial delay in samples (scalar).
        mix_init: initial wet mix in [0, 1] (scalar).
        mode: 0 -> integer delay only, 1 -> linear interpolation.
        delay_smoothing_coeff: smoothing coeff for delay in (0, 1].
        mix_smoothing_coeff: smoothing coeff for mix in (0, 1].
        dtype: JAX dtype for buffer and scalars (default float32).

    Returns:
        state, params
    """
    # Ensure Python ints for shape computation
    max_delay_samples_int = int(max_delay_samples)
    # Choose smallest power-of-two buffer length >= max_delay_samples_int + 1
    # "+1" ensures enough room when delay approaches max.
    buffer_len_pow = 1
    while buffer_len_pow < (max_delay_samples_int + 1):
        buffer_len_pow <<= 1
    buffer_len = buffer_len_pow

    # Allocate buffer outside of any jit context
    buffer = jnp.zeros((buffer_len,), dtype=dtype)

    # Initial write index and smoothed parameters
    write_idx = jnp.array(0, dtype=jnp.int32)
    delay_smooth = jnp.array(delay_init_samples, dtype=dtype)
    mix_smooth = jnp.array(mix_init, dtype=dtype)

    # Pack state and parameters as tuples (no dicts/classes)
    state = (buffer, write_idx, delay_smooth, mix_smooth)

    params = (
        jnp.array(sample_rate, dtype=dtype),                # sample_rate
        jnp.array(max_delay_samples, dtype=dtype),          # max_delay_samples
        jnp.array(buffer_len, dtype=jnp.int32),             # buffer_len
        jnp.array(mode, dtype=jnp.int32),                   # mode
        jnp.array(delay_smoothing_coeff, dtype=dtype),      # delay_smoothing_coeff
        jnp.array(mix_smoothing_coeff, dtype=dtype),        # mix_smoothing_coeff
    )

    return state, params


# -----------------------------------------------------------------------------
# 2. State update helper (e.g. to jump-smoothe params outside audio loop)
# -----------------------------------------------------------------------------


def delay_wetdry_update_state(state, delay_target, mix_target):
    """
    Update state to new delay/mix targets without processing audio.

    This is a utility for externally synchronizing the smoothed parameters
    to specific target values (e.g. for parameter jumps between segments).

    Args:
        state: (buffer, write_idx, delay_smooth, mix_smooth)
        delay_target: scalar or 0D array, new delay in samples
        mix_target: scalar or 0D array, new wet mix in [0, 1]

    Returns:
        new_state with updated delay_smooth and mix_smooth.
    """
    buffer, write_idx, _, _ = state
    delay_smooth = jnp.array(delay_target, dtype=buffer.dtype)
    mix_smooth = jnp.array(mix_target, dtype=buffer.dtype)
    return (buffer, write_idx, delay_smooth, mix_smooth)


# -----------------------------------------------------------------------------
# 3. Single-sample tick
# -----------------------------------------------------------------------------


def _read_delay_sample(buffer, write_idx, delay_samples, mode, max_delay_samples, buffer_len):
    """
    Core delay read with optional integer or linear interpolation.

    Args:
        buffer: [L] delay buffer
        write_idx: scalar int32, index where we will write NEXT (read first!)
        delay_samples: scalar float, current smoothed delay in samples
        mode: int32, 0=integer, 1=linear interpolation
        max_delay_samples: scalar float, maximum delay in samples
        buffer_len: scalar int32, buffer length

    Returns:
        delayed_sample: scalar
    """
    dtype = buffer.dtype

    # Clamp delay
    d_clamped = jnp.clip(delay_samples, jnp.array(0.0, dtype=dtype), max_delay_samples)

    # Compute integer and fractional parts for the linear mode
    d_floor = jnp.floor(d_clamped)
    frac_lin = d_clamped - d_floor

    # Integer mode candidate: round & frac=0
    d_int_mode0 = jnp.round(d_clamped)
    frac_mode0 = jnp.array(0.0, dtype=dtype)

    # Linear mode candidate: floor & actual frac
    d_int_mode1 = d_floor
    frac_mode1 = frac_lin

    # Select based on mode using lax.cond (no Python branching in jit)
    def _select_int(_):
        return d_int_mode0, frac_mode0

    def _select_lin(_):
        return d_int_mode1, frac_mode1

    d_int, frac = lax.cond(
        jnp.equal(mode, jnp.array(0, dtype=mode.dtype)),
        _select_int,
        _select_lin,
        operand=jnp.array(0, dtype=dtype),
    )

    # Wrap read position in the circular buffer
    buffer_len_f = jnp.array(buffer_len, dtype=dtype)
    w_f = jnp.array(write_idx, dtype=dtype)
    r_pos = w_f - d_int
    r_pos_wrapped = jnp.mod(r_pos, buffer_len_f)

    i0_f = jnp.floor(r_pos_wrapped)
    i0 = jnp.mod(i0_f.astype(jnp.int32), buffer_len)
    i1 = jnp.mod(i0 + jnp.array(1, dtype=jnp.int32), buffer_len)

    # Read samples using standard indexing (read-only, differentiable)
    s0 = buffer[i0]
    s1 = buffer[i1]

    # Linear interpolation (frac may be 0 in integer mode)
    one = jnp.array(1.0, dtype=dtype)
    delayed_sample = (one - frac) * s0 + frac * s1
    return delayed_sample


def delay_wetdry_tick(x, delay_target, mix_target, state, params):
    """
    Process a single sample.

    Args:
        x: scalar or 0D array, input sample
        delay_target: scalar, target delay in samples
        mix_target: scalar, target wet mix in [0, 1]
        state: (buffer, write_idx, delay_smooth, mix_smooth)
        params: (sample_rate, max_delay_samples, buffer_len, mode,
                 delay_smoothing_coeff, mix_smoothing_coeff)

    Returns:
        (y, new_state)
    """
    buffer, write_idx, delay_smooth, mix_smooth = state
    (
        sample_rate,
        max_delay_samples,
        buffer_len,
        mode,
        delay_smoothing_coeff,
        mix_smoothing_coeff,
    ) = params

    dtype = buffer.dtype

    # Cast inputs to correct dtype
    x = jnp.array(x, dtype=dtype)
    delay_target = jnp.array(delay_target, dtype=dtype)
    mix_target = jnp.array(mix_target, dtype=dtype)

    # Parameter smoothing (one-pole)
    one = jnp.array(1.0, dtype=dtype)

    delay_smooth_next = delay_smooth + delay_smoothing_coeff * (delay_target - delay_smooth)
    mix_smooth_next = mix_smooth + mix_smoothing_coeff * (mix_target - mix_smooth)

    # Clamp mix to [0, 1]
    mix_smooth_next = jnp.clip(mix_smooth_next, jnp.array(0.0, dtype=dtype), one)

    # Read delayed sample based on updated smoothed delay
    y_delay = _read_delay_sample(
        buffer=buffer,
        write_idx=write_idx,
        delay_samples=delay_smooth_next,
        mode=mode,
        max_delay_samples=max_delay_samples,
        buffer_len=buffer_len,
    )

    # Wet/dry mix
    dry = one - mix_smooth_next
    y = dry * x + mix_smooth_next * y_delay

    # Write current input sample into buffer at write_idx
    x_vec = jnp.reshape(x, (1,))
    buffer_updated = lax.dynamic_update_slice(
        buffer,
        x_vec,
        (write_idx,),
    )

    # Advance write index with wrapping
    write_idx_next = jnp.mod(
        write_idx + jnp.array(1, dtype=jnp.int32),
        buffer_len,
    )

    new_state = (buffer_updated, write_idx_next, delay_smooth_next, mix_smooth_next)
    return y, new_state


# -----------------------------------------------------------------------------
# 4. Block processing via lax.scan
# -----------------------------------------------------------------------------


def delay_wetdry_process(x, delay_target, mix_target, state, params):
    """
    Process a 1D block of samples using lax.scan.

    Args:
        x: [T] array of input samples
        delay_target: [T] or scalar. If scalar, broadcast before calling.
        mix_target: [T] or scalar. If scalar, broadcast before calling.
        state: initial state tuple
        params: params tuple

    Returns:
        y: [T] array of output samples
        final_state: state tuple after processing the block
    """
    x = jnp.asarray(x)

    # Allow scalar delay/mix: broadcast to match x if needed (outside of jit).
    if jnp.ndim(delay_target) == 0:
        delay_target = jnp.ones_like(x) * delay_target
    else:
        delay_target = jnp.asarray(delay_target)
    if jnp.ndim(mix_target) == 0:
        mix_target = jnp.ones_like(x) * mix_target
    else:
        mix_target = jnp.asarray(mix_target)

    def scan_step(carry_state, inputs):
        x_t, d_t, m_t = inputs
        y_t, new_state = delay_wetdry_tick(x_t, d_t, m_t, carry_state, params)
        return new_state, y_t

    xs = (x, delay_target, mix_target)
    final_state, y = lax.scan(scan_step, state, xs)
    return y, final_state


# -----------------------------------------------------------------------------
# 5. Smoke test, plotting, and listening example
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # Basic configuration
    sample_rate = 48000.0
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    max_delay_sec = 0.5
    max_delay_samples = int(sample_rate * max_delay_sec)

    delay_init_sec = 0.2
    delay_init_samples = delay_init_sec * sample_rate

    mix_init = 0.5
    mode = 1  # 0 = integer, 1 = linear interpolation

    # Simple smoothing coefficients
    delay_smoothing_coeff = 0.01
    mix_smoothing_coeff = 0.01

    # Initialize module
    state, params = delay_wetdry_init(
        max_delay_samples=max_delay_samples,
        sample_rate=sample_rate,
        delay_init_samples=delay_init_samples,
        mix_init=mix_init,
        mode=mode,
        delay_smoothing_coeff=delay_smoothing_coeff,
        mix_smoothing_coeff=mix_smoothing_coeff,
        dtype=jnp.float32,
    )

    # Create an impulse input
    x_np = np.zeros(num_samples, dtype=np.float32)
    x_np[0] = 1.0  # single impulse
    x = jnp.array(x_np)

    # Constant delay/mix targets for this demo
    delay_target = jnp.array(delay_init_samples, dtype=jnp.float32)
    mix_target = jnp.array(0.7, dtype=jnp.float32)

    # Optionally JIT the process function
    process_jit = jax.jit(delay_wetdry_process)

    y, final_state = process_jit(x, delay_target, mix_target, state, params)

    y_np = np.array(y)

    # Plot first few milliseconds
    t_ms = np.arange(0, len(y_np)) / sample_rate * 1000.0
    plt.figure(figsize=(10, 4))
    plt.plot(t_ms, y_np, label="Output")
    plt.xlim(0, 50)  # first 50 ms
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("gdelay_wetdry impulse response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen to result if sounddevice is available
    if HAVE_SD:
        print("Playing delayed impulse...")
        sd.play(y_np, int(sample_rate))
        sd.wait()
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")

"""
Follow-up prompts you could use:

1. “Modify this delay module to add a feedback parameter so it becomes a comb filter, while keeping everything differentiable and JAX-friendly.”

2. “Extend gdelay_wetdry to support stereo processing (2-channel) with a shared delay buffer or separate buffers per channel; keep the same functional API style.”

3. “Add a modulation input (e.g., an LFO) to modulate the delay time smoothly and update the code so that modulation remains differentiable and stable.”

4. “Refactor this module to support multiple taps (multitap delay) sharing a single buffer, with independent wet mixes per tap.”

5. “Show how to backpropagate through gdelay_wetdry to learn a time-varying delay curve that transforms one signal into another using JAX autodiff.”
"""
```
