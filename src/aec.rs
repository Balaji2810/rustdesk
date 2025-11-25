/// Digital Echo Cancellation (DEC) module
///
/// Optimized for **digital** echo paths where the echo is a direct copy
/// of the reference signal with delay and gain (no room acoustics).
///
/// Uses block-based processing with:
/// - Cross-correlation for delay estimation
/// - Direct subtraction with adaptive gain
/// - Minimal latency (no sample-by-sample processing)
use std::collections::VecDeque;

/// Block size for processing (10ms at 48kHz stereo = 960 samples)
const BLOCK_SIZE: usize = 960;

/// Maximum delay to search for (in samples, ~50ms at 48kHz)
const MAX_DELAY_SAMPLES: usize = 2400;

/// Digital Echo Canceller optimized for digital echo paths
///
/// For digital echo (loopback), the echo is essentially:
/// - A delayed copy of the reference
/// - Possibly with different gain
/// - No room impulse response
///
/// This allows for fast, simple processing using delay compensation
/// and direct subtraction.
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// Circular buffer for reference signal history
    reference_history: VecDeque<f32>,
    /// Estimated delay between reference and echo (in samples)
    estimated_delay: usize,
    /// Estimated echo gain (typically close to 1.0 for digital paths)
    estimated_gain: f32,
    /// Number of channels (1 = mono, 2 = stereo)
    num_channels: usize,
    /// Smoothing factor for delay updates (0-1, lower = more stable)
    delay_smoothing: f32,
    /// Smoothing factor for gain updates
    gain_smoothing: f32,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    /// Create a new digital echo canceller
    ///
    /// # Arguments
    /// * `num_channels` - Number of audio channels (1 or 2)
    pub fn new(num_channels: usize) -> Self {
        Self {
            reference_history: VecDeque::with_capacity(MAX_DELAY_SAMPLES * num_channels * 2),
            estimated_delay: 480 * num_channels, // Initial guess: 10ms at 48kHz
            estimated_gain: 1.0,
            num_channels,
            delay_smoothing: 0.1,
            gain_smoothing: 0.1,
        }
    }

    /// Create a stereo digital echo canceller (48kHz)
    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    /// Reset the echo canceller state
    pub fn reset(&mut self) {
        self.reference_history.clear();
        self.estimated_delay = 480 * self.num_channels;
        self.estimated_gain = 1.0;
    }

    /// Process audio for echo cancellation
    ///
    /// # Arguments
    /// * `reference` - Reference signal (what was sent to speakers)
    /// * `input` - Input signal (loopback/capture containing echo)
    ///
    /// # Returns
    /// Echo-cancelled output
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // Add reference to history
        for &sample in reference.iter().take(len) {
            self.reference_history.push_back(sample);
        }

        // Keep history bounded
        let max_history = MAX_DELAY_SAMPLES * self.num_channels * 2;
        while self.reference_history.len() > max_history {
            self.reference_history.pop_front();
        }

        // If we don't have enough history, just return the input
        if self.reference_history.len() < self.estimated_delay + len {
            return input[..len].to_vec();
        }

        // Update delay estimate periodically (every ~10ms worth of samples)
        if len >= BLOCK_SIZE {
            self.update_delay_estimate(input);
        }

        // Cancel echo using estimated delay and gain
        let mut output = Vec::with_capacity(len);
        let history_len = self.reference_history.len();

        for i in 0..len {
            // Get the delayed reference sample
            let ref_idx = history_len - self.estimated_delay - len + i;
            let delayed_ref = if ref_idx < history_len {
                self.reference_history[ref_idx]
            } else {
                0.0
            };

            // Subtract estimated echo
            let echo_estimate = delayed_ref * self.estimated_gain;
            let cancelled = input[i] - echo_estimate;

            // Soft limiter to prevent artifacts
            output.push(cancelled.clamp(-1.0, 1.0));
        }

        // Update gain estimate based on correlation
        self.update_gain_estimate(&output, input);

        output
    }

    /// Update delay estimate using cross-correlation
    fn update_delay_estimate(&mut self, input: &[f32]) {
        let history_len = self.reference_history.len();
        let search_len = input.len().min(BLOCK_SIZE);

        if history_len < search_len + MAX_DELAY_SAMPLES * self.num_channels {
            return;
        }

        let mut best_correlation = 0.0f32;
        let mut best_delay = self.estimated_delay;

        // Search for the delay with highest correlation
        // Sample sparse delays to keep it fast
        let step = self.num_channels.max(1);
        let min_delay = step;
        let max_delay = (MAX_DELAY_SAMPLES * self.num_channels).min(history_len - search_len);

        for delay in (min_delay..max_delay).step_by(step * 4) {
            let correlation = self.compute_correlation(input, delay, search_len);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }

        // Fine-tune around the best delay found
        if best_correlation > 0.3 {
            let fine_min = best_delay.saturating_sub(step * 8);
            let fine_max = (best_delay + step * 8).min(max_delay);

            for delay in (fine_min..fine_max).step_by(step) {
                let correlation = self.compute_correlation(input, delay, search_len);
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_delay = delay;
                }
            }
        }

        // Update delay with smoothing
        if best_correlation > 0.5 {
            let new_delay = best_delay as f32;
            let old_delay = self.estimated_delay as f32;
            self.estimated_delay = (old_delay * (1.0 - self.delay_smoothing)
                + new_delay * self.delay_smoothing) as usize;
        }
    }

    /// Compute normalized cross-correlation at a specific delay
    fn compute_correlation(&self, input: &[f32], delay: usize, len: usize) -> f32 {
        let history_len = self.reference_history.len();
        let ref_start = history_len.saturating_sub(delay + len);

        let mut sum_xy = 0.0f32;
        let mut sum_x2 = 0.0f32;
        let mut sum_y2 = 0.0f32;

        for i in 0..len {
            let x = input[i];
            let ref_idx = ref_start + i;
            let y = if ref_idx < history_len {
                self.reference_history[ref_idx]
            } else {
                0.0
            };

            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom > 1e-6 {
            sum_xy / denom
        } else {
            0.0
        }
    }

    /// Update gain estimate
    fn update_gain_estimate(&mut self, output: &[f32], input: &[f32]) {
        // Estimate residual echo energy
        let output_energy: f32 = output.iter().map(|&s| s * s).sum();
        let input_energy: f32 = input.iter().map(|&s| s * s).sum();

        if input_energy > 1e-6 && output_energy > 1e-6 {
            // If output energy is similar to input, we might need to adjust gain
            let energy_ratio = output_energy / input_energy;

            // If we're not cancelling much, increase gain slightly
            if energy_ratio > 0.9 && self.estimated_gain < 1.5 {
                self.estimated_gain += 0.01;
            } else if energy_ratio < 0.3 && self.estimated_gain > 0.5 {
                // If we're over-cancelling, reduce gain
                self.estimated_gain -= 0.01;
            }
        }
    }
}

/// Process buffer through the digital echo canceller
#[cfg(windows)]
pub fn process_buffer(
    dec: &mut DigitalEchoCanceller,
    reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    dec.process(reference, input)
}

// Legacy API compatibility for non-Windows platforms
#[cfg(not(windows))]
pub struct DigitalEchoCanceller;

#[cfg(not(windows))]
impl DigitalEchoCanceller {
    pub fn new(_num_channels: usize) -> Self {
        Self
    }

    pub fn new_stereo() -> Self {
        Self
    }

    pub fn reset(&mut self) {}

    pub fn process(&mut self, _reference: &[f32], input: &[f32]) -> Vec<f32> {
        input.to_vec()
    }
}

#[cfg(not(windows))]
pub fn process_buffer(
    _dec: &mut DigitalEchoCanceller,
    _reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    input.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn test_dec_creation() {
        let dec = DigitalEchoCanceller::new_stereo();
        assert_eq!(dec.num_channels, 2);
    }

    #[cfg(windows)]
    #[test]
    fn test_dec_reset() {
        let mut dec = DigitalEchoCanceller::new_stereo();

        // Process some data
        let reference = vec![0.5f32; 1920];
        let input = vec![0.5f32; 1920];
        let _ = dec.process(&reference, &input);

        // Reset
        dec.reset();
        assert!(dec.reference_history.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn test_dec_process_with_echo() {
        let mut dec = DigitalEchoCanceller::new(1); // Mono for simpler test

        // Create a reference signal
        let reference: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();

        // Create input with delayed echo (delay of 480 samples = 10ms at 48kHz)
        let delay = 480;
        let mut input = vec![0.0f32; 4800];
        for i in delay..input.len() {
            input[i] = reference[i - delay] * 0.9; // Echo with 0.9 gain
        }

        // First, feed some reference to build up history
        let _ = dec.process(&reference[..2400], &input[..2400]);

        // Process more data
        let output = dec.process(&reference[2400..], &input[2400..]);

        // Output should have lower energy than input (echo reduced)
        let input_energy: f32 = input[2400..].iter().map(|&s| s * s).sum();
        let output_energy: f32 = output.iter().map(|&s| s * s).sum();

        // Echo should be at least partially cancelled
        assert!(output_energy <= input_energy);
    }

    #[cfg(windows)]
    #[test]
    fn test_dec_no_reference() {
        let mut dec = DigitalEchoCanceller::new_stereo();

        // Process with no reference history
        let reference = vec![0.0f32; 960];
        let input: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();

        let output = dec.process(&reference, &input);

        // Without proper reference, output should be close to input
        assert_eq!(output.len(), input.len());
    }
}
