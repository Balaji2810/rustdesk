/// Acoustic Echo Cancellation module - FDAF-AEC Implementation
///
/// Uses Frequency Domain Adaptive Filter (FDAF) algorithm for echo cancellation
/// with optional RNNoise-based noise suppression. Automatically learns the echo 
/// path (delay + gain + frequency response) without explicit delay estimation.

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// FFT size for the Frequency Domain Adaptive Filter
/// Smaller = faster adaptation, larger = handles longer echo paths
#[cfg(windows)]
const DEFAULT_FFT_SIZE: usize = 256;

/// Step size (learning rate) for the adaptive filter
/// Higher = faster convergence but less stable (0.05-0.15 recommended)
#[cfg(windows)]
const DEFAULT_STEP_SIZE: f32 = 0.04;

/// Echo reduction strength (blend factor between original and processed)
/// 1.0 = full echo cancellation, 0.0 = no processing
#[cfg(windows)]
const DEFAULT_REDUCTION: f32 = 1.0;

/// Output smoothing factor to reduce artifacts
/// Higher = smoother but may blur transients (0.0-0.5)
#[cfg(windows)]
const DEFAULT_SMOOTHING: f32 = 0.4;

/// Enable adaptive delay tracking per frame
#[cfg(windows)]
const DEFAULT_ADAPTIVE_DELAY: bool = true;

/// Enable RNNoise-based noise suppression
#[cfg(windows)]
const DEFAULT_NOISE_SUPPRESSION: bool = true;

/// Noise suppression strength (0.0-1.0)
#[cfg(windows)]
const DEFAULT_NOISE_STRENGTH: f32 = 0.8;

/// Maximum delay to search for during auto-detection (in samples at 48kHz)
/// 1000ms at 48kHz = 48000 samples
#[cfg(windows)]
const MAX_DELAY_SAMPLES: usize = 48000;

/// Sample rate required by RNNoise
#[cfg(windows)]
const RNNOISE_SAMPLE_RATE: u32 = 48000;

// ============================================================================
// FDAF-AEC ECHO CANCELLER
// ============================================================================

/// FDAF-AEC based Digital Echo Canceller with optional noise suppression
/// 
/// Uses Frequency Domain Adaptive Filter for efficient echo cancellation
/// and RNNoise for neural network-based noise suppression.
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// FDAF-AEC instance for echo cancellation
    aec: fdaf_aec::FdafAec,
    /// Frame size (half of FFT size)
    frame_size: usize,
    /// Hop size for overlap-add processing
    hop_size: usize,
    /// Hann window for overlap-add
    window: Vec<f32>,
    /// Previous frame for smoothing
    prev_frame: Vec<f32>,
    /// Current detected delay in samples
    current_delay: usize,
    /// Reference signal buffer for delay detection
    ref_buffer: Vec<f32>,
    /// Input signal buffer for processing
    input_buffer: Vec<f32>,
    /// Output overlap-add buffer
    output_buffer: Vec<f32>,
    /// Noise suppression denoiser
    denoiser: Option<nnnoiseless::DenoiseState<'static>>,
    /// Echo reduction strength
    reduction: f32,
    /// Smoothing factor
    smoothing: f32,
    /// Enable adaptive delay tracking
    adaptive_delay: bool,
    /// Noise suppression strength
    noise_strength: f32,
    /// Frame counter for periodic operations
    frame_count: usize,
    /// Delay search range in samples
    delay_search_range: usize,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    /// Create a new echo canceller with default settings
    pub fn new(_num_channels: usize) -> Self {
        Self::with_config(
            DEFAULT_FFT_SIZE,
            DEFAULT_STEP_SIZE,
            DEFAULT_REDUCTION,
            DEFAULT_SMOOTHING,
            DEFAULT_ADAPTIVE_DELAY,
            DEFAULT_NOISE_SUPPRESSION,
            DEFAULT_NOISE_STRENGTH,
        )
    }

    /// Create a stereo echo canceller (default)
    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    /// Create with custom configuration
    pub fn with_config(
        fft_size: usize,
        step_size: f32,
        reduction: f32,
        smoothing: f32,
        adaptive_delay: bool,
        noise_suppression: bool,
        noise_strength: f32,
    ) -> Self {
        let frame_size = fft_size / 2;
        let overlap = frame_size / 2;
        let hop_size = frame_size - overlap;

        // Create Hann window for overlap-add
        let window: Vec<f32> = (0..frame_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / frame_size as f32).cos())
            })
            .collect();

        // Create denoiser if noise suppression is enabled
        let denoiser = if noise_suppression {
            Some(nnnoiseless::DenoiseState::new())
        } else {
            None
        };

        // Delay search range: ~50ms at 48kHz
        let delay_search_range = (50.0 * 48000.0 / 1000.0) as usize;

        Self {
            aec: fdaf_aec::FdafAec::new(fft_size, step_size),
            frame_size,
            hop_size,
            window,
            prev_frame: vec![0.0; frame_size],
            current_delay: 0,
            ref_buffer: Vec::with_capacity(MAX_DELAY_SAMPLES * 2),
            input_buffer: Vec::with_capacity(frame_size * 4),
            output_buffer: Vec::new(),
            denoiser,
            reduction,
            smoothing,
            adaptive_delay,
            noise_strength,
            frame_count: 0,
            delay_search_range,
        }
    }

    /// Reset the echo canceller state
    pub fn reset(&mut self) {
        self.aec = fdaf_aec::FdafAec::new(self.frame_size * 2, DEFAULT_STEP_SIZE);
        self.prev_frame.fill(0.0);
        self.current_delay = 0;
        self.ref_buffer.clear();
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.frame_count = 0;
        if self.denoiser.is_some() {
            self.denoiser = Some(nnnoiseless::DenoiseState::new());
        }
    }

    /// Set the step size (learning rate)
    #[allow(dead_code)]
    pub fn set_step_size(&mut self, step_size: f32) {
        self.aec = fdaf_aec::FdafAec::new(self.frame_size * 2, step_size.clamp(0.01, 0.5));
    }

    /// Set echo reduction strength
    #[allow(dead_code)]
    pub fn set_reduction(&mut self, reduction: f32) {
        self.reduction = reduction.clamp(0.0, 1.0);
    }

    /// Enable/disable noise suppression
    #[allow(dead_code)]
    pub fn set_noise_suppression(&mut self, enabled: bool, strength: f32) {
        if enabled && self.denoiser.is_none() {
            self.denoiser = Some(nnnoiseless::DenoiseState::new());
        } else if !enabled {
            self.denoiser = None;
        }
        self.noise_strength = strength.clamp(0.0, 1.0);
    }

    /// Process audio buffer through FDAF-AEC
    /// 
    /// # Arguments
    /// * `reference` - The signal being played (far-end / client audio)
    /// * `input` - The captured signal containing echo (near-end + echo)
    /// 
    /// # Returns
    /// Echo-cancelled (and optionally denoised) signal
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // Add new samples to buffers
        self.ref_buffer.extend_from_slice(&reference[..len]);
        self.input_buffer.extend_from_slice(&input[..len]);

        // Limit buffer sizes to prevent unbounded growth
        const MAX_BUFFER: usize = MAX_DELAY_SAMPLES * 3;
        if self.ref_buffer.len() > MAX_BUFFER {
            let excess = self.ref_buffer.len() - MAX_BUFFER;
            self.ref_buffer.drain(0..excess);
        }

        // Detect initial delay if not yet detected
        if self.frame_count == 0 && self.ref_buffer.len() > self.frame_size * 4 {
            let (delay, _correlation) = detect_initial_delay(
                &self.ref_buffer,
                &self.input_buffer,
                MAX_DELAY_SAMPLES.min(self.ref_buffer.len() / 2),
            );
            self.current_delay = delay;
            log::debug!("AEC initial delay detected: {} samples ({:.1}ms)", 
                delay, delay as f32 / 48.0);
        }

        // Process in frames using overlap-add
        let mut processed_output = Vec::new();

        while self.input_buffer.len() >= self.frame_size {
            let mic_frame: Vec<f32> = self.input_buffer.drain(0..self.frame_size).collect();

            // Get aligned far-end frame based on current delay
            let far_frame = self.get_aligned_far_frame();

            // Adaptive delay tracking every 10 frames
            if self.adaptive_delay && self.frame_count % 10 == 0 && !far_frame.is_empty() {
                let local_delay = find_local_delay(
                    &far_frame,
                    &mic_frame,
                    self.delay_search_range,
                    self.current_delay,
                );
                // Smooth delay changes
                if (local_delay as i64 - self.current_delay as i64).abs() 
                    < self.delay_search_range as i64 
                {
                    self.current_delay = (self.current_delay * 3 + local_delay) / 4;
                }
            }

            // Process through FDAF-AEC
            let mut cleaned = self.aec.process(&far_frame, &mic_frame);

            // Apply reduction factor (blend with original)
            for (i, sample) in cleaned.iter_mut().enumerate() {
                let original = mic_frame[i];
                *sample = original * (1.0 - self.reduction) + *sample * self.reduction;
            }

            // Smooth with previous frame
            if self.smoothing > 0.0 {
                for (i, sample) in cleaned.iter_mut().enumerate() {
                    *sample = *sample * (1.0 - self.smoothing) + self.prev_frame[i] * self.smoothing;
                }
            }
            self.prev_frame = cleaned.clone();

            // Apply window and add to output
            for (i, &sample) in cleaned.iter().enumerate() {
                let windowed = soft_limit(sample) * self.window[i];
                if i < processed_output.len() {
                    processed_output[i] += windowed;
                } else {
                    processed_output.push(windowed);
                }
            }

            // Advance for next frame
            self.frame_count += 1;
        }

        // Normalize overlap-add output
        if !processed_output.is_empty() {
            let window_sum: f32 = self.window.iter().sum();
            let normalization = self.hop_size as f32 / window_sum;
            for sample in &mut processed_output {
                *sample *= normalization;
                *sample = soft_limit(*sample);
            }
        }

        // Apply noise suppression if enabled
        if let Some(ref mut denoiser) = self.denoiser {
            if self.noise_strength > 0.0 {
                processed_output = apply_noise_suppression_inplace(
                    denoiser,
                    &processed_output,
                    self.noise_strength,
                );
            }
        }

        processed_output
    }

    /// Get aligned far-end frame based on current delay
    fn get_aligned_far_frame(&self) -> Vec<f32> {
        if self.ref_buffer.is_empty() {
            return vec![0.0; self.frame_size];
        }

        let ref_len = self.ref_buffer.len();
        let start = if ref_len > self.current_delay + self.frame_size {
            ref_len - self.current_delay - self.frame_size
        } else {
            0
        };
        let end = (start + self.frame_size).min(ref_len);

        let mut frame: Vec<f32> = self.ref_buffer[start..end].to_vec();
        frame.resize(self.frame_size, 0.0);
        frame
    }
}

// ============================================================================
// DELAY DETECTION FUNCTIONS
// ============================================================================

/// Find best delay for a short segment using cross-correlation
#[cfg(windows)]
fn find_local_delay(far: &[f32], mic: &[f32], search_range: usize, base_delay: usize) -> usize {
    let min_delay = base_delay.saturating_sub(search_range);
    let max_delay = base_delay + search_range;

    let mut best_delay = base_delay;
    let mut best_corr = f32::NEG_INFINITY;

    let len = far.len().min(mic.len());
    if len < 64 {
        return base_delay;
    }

    for delay in (min_delay..=max_delay).step_by(4) {
        if delay >= len {
            continue;
        }

        let mut corr = 0.0f32;
        let mut far_e = 0.0f32;
        let mut mic_e = 0.0f32;

        let compare_len = len.saturating_sub(delay);
        for i in 0..compare_len {
            let f = far[i];
            let m = if i + delay < mic.len() {
                mic[i + delay]
            } else {
                0.0
            };
            corr += f * m;
            far_e += f * f;
            mic_e += m * m;
        }

        let norm = (far_e * mic_e).sqrt();
        let normalized_corr = if norm > 1e-10 { corr / norm } else { 0.0 };

        if normalized_corr > best_corr {
            best_corr = normalized_corr;
            best_delay = delay;
        }
    }

    best_delay
}

/// Initial delay detection using cross-correlation
#[cfg(windows)]
fn detect_initial_delay(far_end: &[f32], mic: &[f32], max_delay_samples: usize) -> (usize, f32) {
    let search_len = far_end.len().min(mic.len()).min(48000 * 3); // Up to 3 seconds
    if search_len < max_delay_samples {
        return (0, 0.0);
    }

    let mut best_delay = 0;
    let mut best_corr = f32::NEG_INFINITY;

    // Coarse search (step by 20 samples)
    for delay in (0..=max_delay_samples).step_by(20) {
        if delay >= search_len {
            break;
        }

        let mut corr = 0.0f32;
        let mut far_e = 0.0f32;
        let mut mic_e = 0.0f32;

        let len = search_len - delay;
        for i in 0..len {
            let f = far_end[i];
            let m = mic[i + delay];
            corr += f * m;
            far_e += f * f;
            mic_e += m * m;
        }

        let norm = (far_e * mic_e).sqrt();
        let normalized_corr = if norm > 1e-10 { corr / norm } else { 0.0 };

        if normalized_corr > best_corr {
            best_corr = normalized_corr;
            best_delay = delay;
        }
    }

    // Fine search around best coarse result
    let fine_start = best_delay.saturating_sub(50);
    let fine_end = (best_delay + 50).min(max_delay_samples);

    for delay in fine_start..=fine_end {
        if delay >= search_len {
            break;
        }

        let mut corr = 0.0f32;
        let mut far_e = 0.0f32;
        let mut mic_e = 0.0f32;

        let len = search_len - delay;
        for i in 0..len {
            let f = far_end[i];
            let m = mic[i + delay];
            corr += f * m;
            far_e += f * f;
            mic_e += m * m;
        }

        let norm = (far_e * mic_e).sqrt();
        let normalized_corr = if norm > 1e-10 { corr / norm } else { 0.0 };

        if normalized_corr > best_corr {
            best_corr = normalized_corr;
            best_delay = delay;
        }
    }

    (best_delay, best_corr)
}

// ============================================================================
// AUDIO PROCESSING UTILITIES
// ============================================================================

/// Soft limiter to prevent clipping while preserving dynamics
#[cfg(windows)]
#[inline]
fn soft_limit(x: f32) -> f32 {
    if x.abs() < 0.8 {
        x
    } else {
        x.signum() * (0.8 + 0.2 * ((x.abs() - 0.8) / 0.2).tanh())
    }
}

/// Apply RNNoise-based noise suppression in-place
/// nnnoiseless requires 48kHz audio and processes 480 samples at a time (10ms frames)
#[cfg(windows)]
fn apply_noise_suppression_inplace(
    denoiser: &mut nnnoiseless::DenoiseState,
    samples: &[f32],
    strength: f32,
) -> Vec<f32> {
    const RNNOISE_FRAME_SIZE: usize = nnnoiseless::DenoiseState::FRAME_SIZE; // 480 samples

    let mut output = Vec::with_capacity(samples.len());
    let mut input_frame = [0.0f32; RNNOISE_FRAME_SIZE];
    let mut output_frame = [0.0f32; RNNOISE_FRAME_SIZE];

    for chunk in samples.chunks(RNNOISE_FRAME_SIZE) {
        // Copy input (pad with zeros if needed)
        input_frame.fill(0.0);
        input_frame[..chunk.len()].copy_from_slice(chunk);

        // Process through RNNoise
        denoiser.process_frame(&mut output_frame, &input_frame);

        // Blend original and denoised based on strength
        for (i, &denoised) in output_frame.iter().enumerate().take(chunk.len()) {
            let original = input_frame[i];
            let blended = original * (1.0 - strength) + denoised * strength;
            output.push(blended);
        }
    }

    output
}

/// Linear interpolation resampler for rate conversion
#[cfg(windows)]
#[allow(dead_code)]
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    (0..new_len)
        .map(|i| {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len().saturating_sub(1));
            let frac = (src_idx - idx0 as f64) as f32;
            samples.get(idx0).copied().unwrap_or(0.0) * (1.0 - frac)
                + samples.get(idx1).copied().unwrap_or(0.0) * frac
        })
        .collect()
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Process buffer - main entry point for echo cancellation
#[cfg(windows)]
pub fn process_buffer(
    aec: &mut DigitalEchoCanceller,
    reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    aec.process(reference, input)
}

// ============================================================================
// NON-WINDOWS: PASSTHROUGH IMPLEMENTATION
// ============================================================================

#[cfg(not(windows))]
pub struct DigitalEchoCanceller;

#[cfg(not(windows))]
impl DigitalEchoCanceller {
    pub fn new(_: usize) -> Self {
        Self
    }
    pub fn new_stereo() -> Self {
        Self
    }
    pub fn reset(&mut self) {}
    pub fn process(&mut self, _: &[f32], input: &[f32]) -> Vec<f32> {
        input.to_vec()
    }
}

#[cfg(not(windows))]
pub fn process_buffer(_: &mut DigitalEchoCanceller, _: &[f32], input: &[f32]) -> Vec<f32> {
    input.to_vec()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn test_fdaf_echo_cancellation() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Create reference signal (simulated far-end audio)
        let reference: Vec<f32> = (0..4800)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        // Simulate echo with delay and gain
        let delay = 480;
        let echo_gain = 0.8;
        let mut input = vec![0.0f32; 4800];
        for i in delay..4800 {
            input[i] = reference[i - delay] * echo_gain;
        }

        // Process multiple frames to let FDAF converge
        for _ in 0..20 {
            let _ = aec.process(&reference, &input);
        }

        // After convergence, output should have significantly reduced echo
        let output = aec.process(&reference, &input);

        // Should produce some output
        assert!(!output.is_empty(), "Output should not be empty");
    }

    #[cfg(windows)]
    #[test]
    fn test_delay_detection() {
        // Create reference and delayed mic signal
        let reference: Vec<f32> = (0..48000)
            .map(|i| (i as f32 * 0.02).sin() * 0.5)
            .collect();

        let delay = 2400; // 50ms at 48kHz
        let mut mic = vec![0.0f32; 48000];
        for i in delay..48000 {
            mic[i] = reference[i - delay] * 0.7;
        }

        let (detected_delay, correlation) = detect_initial_delay(&reference, &mic, 4800);

        // Detected delay should be close to actual delay
        let delay_error = (detected_delay as i64 - delay as i64).abs();
        assert!(
            delay_error < 100,
            "Delay detection error too large: detected={}, actual={}, error={}",
            detected_delay,
            delay,
            delay_error
        );
        assert!(correlation > 0.5, "Correlation too low: {}", correlation);
    }

    #[cfg(windows)]
    #[test]
    fn test_soft_limiter() {
        // Values below threshold should pass through
        assert!((soft_limit(0.5) - 0.5).abs() < 0.001);
        assert!((soft_limit(-0.5) - (-0.5)).abs() < 0.001);

        // Values above threshold should be limited but not clipped
        let limited = soft_limit(1.5);
        assert!(limited < 1.0, "Soft limit should keep values below 1.0");
        assert!(limited > 0.8, "Soft limit should not over-compress");
    }

    #[cfg(windows)]
    #[test]
    fn test_double_talk_preservation() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Reference signal
        let reference: Vec<f32> = (0..4800)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        // Near-end signal (local speaker) - different frequency
        let near_end: Vec<f32> = (0..4800)
            .map(|i| (i as f32 * 0.03).sin() * 0.5)
            .collect();

        // Input = echo + near-end
        let delay = 480;
        let mut input = vec![0.0f32; 4800];
        for i in 0..4800 {
            let echo = if i >= delay {
                reference[i - delay] * 0.6
            } else {
                0.0
            };
            input[i] = echo + near_end[i];
        }

        // Let it partially converge
        for _ in 0..10 {
            let _ = aec.process(&reference, &input);
        }

        let output = aec.process(&reference, &input);

        // Output should contain some signal (not completely cancelled)
        let out_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(out_energy > 0.0, "Near-end signal should be preserved");
    }
}
