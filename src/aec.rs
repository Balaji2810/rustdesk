/// Digital Echo Cancellation module - Optimized for loopback audio
///
/// For digital loopback (like RustDesk), the echo is a near-perfect delayed
/// copy of the reference. This implementation uses correlation-based delay
/// finding with simple subtraction - much more stable than adaptive filters.

use std::collections::VecDeque;

/// Maximum delay to search (samples) - ~100ms at 48kHz stereo
const MAX_DELAY: usize = 9600;

/// Minimum delay to search (samples) - ~1ms at 48kHz stereo  
const MIN_DELAY: usize = 48;

/// Correlation window size for delay estimation
const CORR_WINDOW: usize = 2400;

/// How often to re-estimate delay (in process calls)
const ESTIMATE_INTERVAL: usize = 100;

/// Digital Echo Canceller - optimized for loopback scenarios
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// Reference signal ring buffer
    ref_buffer: VecDeque<f32>,
    /// Estimated delay in samples
    delay: usize,
    /// Estimated gain (echo amplitude relative to reference)
    gain: f32,
    /// Counter for periodic delay re-estimation
    frame_count: usize,
    /// Whether we've found a valid delay
    delay_locked: bool,
    /// Smoothed correlation strength
    correlation_strength: f32,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(_num_channels: usize) -> Self {
        Self {
            ref_buffer: VecDeque::with_capacity(MAX_DELAY + CORR_WINDOW),
            delay: 960, // Initial guess: ~10ms at 48kHz stereo
            gain: 0.0,  // Start with no cancellation until we lock
            frame_count: 0,
            delay_locked: false,
            correlation_strength: 0.0,
        }
    }

    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    pub fn reset(&mut self) {
        self.ref_buffer.clear();
        self.delay = 960;
        self.gain = 0.0;
        self.frame_count = 0;
        self.delay_locked = false;
        self.correlation_strength = 0.0;
    }

    /// Process audio - find delay and subtract echo
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // Add reference samples to buffer
        for &s in reference.iter().take(len) {
            if self.ref_buffer.len() >= MAX_DELAY + CORR_WINDOW {
                self.ref_buffer.pop_front();
            }
            self.ref_buffer.push_back(s);
        }

        // Not enough history - pass through unchanged
        if self.ref_buffer.len() < MAX_DELAY + len {
            return input[..len].to_vec();
        }

        // Periodically re-estimate delay
        self.frame_count += 1;
        if self.frame_count >= ESTIMATE_INTERVAL || !self.delay_locked {
            if len >= CORR_WINDOW / 2 {
                self.estimate_delay_and_gain(input);
            }
        }

        // If we haven't locked onto a delay yet, pass through
        if !self.delay_locked || self.gain < 0.1 {
            return input[..len].to_vec();
        }

        // Cancel echo: output = input - gain * delayed_reference
        let buf_len = self.ref_buffer.len();
        let ref_start = buf_len.saturating_sub(self.delay + len);

        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let ref_sample = self.ref_buffer.get(ref_start + i).copied().unwrap_or(0.0);
            // Subtract with a safety factor to avoid over-cancellation
            let cancelled = input[i] - ref_sample * self.gain * 0.9;
            output.push(cancelled.clamp(-1.0, 1.0));
        }

        output
    }

    /// Estimate delay using normalized cross-correlation
    fn estimate_delay_and_gain(&mut self, input: &[f32]) {
        let buf_len = self.ref_buffer.len();
        if buf_len < CORR_WINDOW + MAX_DELAY {
            return;
        }

        let window = CORR_WINDOW.min(input.len());
        if window < 240 {
            return;
        }

        // Calculate input signal energy
        let input_energy: f32 = input[..window].iter().map(|x| x * x).sum();
        if input_energy < 1e-6 {
            return; // No signal in input
        }

        let mut best_corr = 0.0f32;
        let mut best_delay = self.delay;
        let mut best_gain = self.gain;

        // Search for best delay using normalized cross-correlation
        // Use coarse search first, then fine search
        let step = if self.delay_locked { 4 } else { 16 };
        
        for delay in (MIN_DELAY..MAX_DELAY).step_by(step) {
            let ref_start = buf_len - delay - window;
            
            let mut sum_xy = 0.0f32;
            let mut sum_yy = 0.0f32;
            
            for i in 0..window {
                let x = input[i];
                let y = self.ref_buffer.get(ref_start + i).copied().unwrap_or(0.0);
                sum_xy += x * y;
                sum_yy += y * y;
            }
            
            // Skip if reference has no energy at this delay
            if sum_yy < 1e-6 {
                continue;
            }
            
            // Normalized correlation
            let corr = sum_xy / (input_energy * sum_yy).sqrt();
            
            if corr > best_corr {
                best_corr = corr;
                best_delay = delay;
                // Optimal gain is sum_xy / sum_yy (least squares)
                best_gain = (sum_xy / sum_yy).clamp(0.0, 2.0);
            }
        }

        // Fine search around best delay (if we found something promising)
        if best_corr > 0.3 {
            let search_start = best_delay.saturating_sub(step * 2).max(MIN_DELAY);
            let search_end = (best_delay + step * 2).min(MAX_DELAY);
            
            for delay in search_start..search_end {
                let ref_start = buf_len - delay - window;
                
                let mut sum_xy = 0.0f32;
                let mut sum_yy = 0.0f32;
                
                for i in 0..window {
                    let x = input[i];
                    let y = self.ref_buffer.get(ref_start + i).copied().unwrap_or(0.0);
                    sum_xy += x * y;
                    sum_yy += y * y;
                }
                
                if sum_yy < 1e-6 {
                    continue;
                }
                
                let corr = sum_xy / (input_energy * sum_yy).sqrt();
                
                if corr > best_corr {
                    best_corr = corr;
                    best_delay = delay;
                    best_gain = (sum_xy / sum_yy).clamp(0.0, 2.0);
                }
            }
        }

        // Update with smoothing
        self.correlation_strength = self.correlation_strength * 0.8 + best_corr * 0.2;

        // Only update if correlation is strong enough
        if best_corr > 0.5 {
            if self.delay_locked {
                // Smooth update when locked
                self.delay = (self.delay * 3 + best_delay) / 4;
                self.gain = self.gain * 0.8 + best_gain * 0.2;
            } else {
                // First lock - use values directly
                self.delay = best_delay;
                self.gain = best_gain;
                self.delay_locked = true;
            }
            self.frame_count = 0;
        } else if self.correlation_strength < 0.2 {
            // Lost correlation - reduce gain gradually
            self.gain *= 0.95;
            if self.gain < 0.1 {
                self.delay_locked = false;
            }
        }
    }
}

/// Process buffer - main entry point
#[cfg(windows)]
pub fn process_buffer(
    dec: &mut DigitalEchoCanceller,
    reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    dec.process(reference, input)
}

// Non-Windows: passthrough
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn test_passthrough_no_echo() {
        let mut dec = DigitalEchoCanceller::new_stereo();

        // When there's no correlation, should pass through
        let reference: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin()).collect();
        let input: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.03).cos()).collect(); // Different signal
        
        // Process a few times
        for _ in 0..5 {
            let output = dec.process(&reference, &input);
            // Should be close to input since no echo detected
            let diff: f32 = output.iter()
                .zip(input.iter())
                .map(|(o, i)| (o - i).abs())
                .sum::<f32>() / output.len() as f32;
            assert!(diff < 0.5, "Output differs too much: {}", diff);
        }
    }

    #[cfg(windows)]
    #[test]
    fn test_echo_cancellation() {
        let mut dec = DigitalEchoCanceller::new_stereo();

        // Create reference and echoed input
        let reference: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let delay = 480;
        let echo_gain = 0.8;
        let mut input = vec![0.0f32; 4800];
        for i in delay..4800 {
            input[i] = reference[i - delay] * echo_gain;
        }

        // Process multiple times to find delay
        for _ in 0..10 {
            let _ = dec.process(&reference, &input);
        }

        assert!(dec.delay_locked, "Should have locked onto delay");

        let output = dec.process(&reference, &input);

        // Output should have reduced energy
        let measure_start = delay + 200;
        let in_energy: f32 = input[measure_start..].iter().map(|x| x * x).sum();
        let out_energy: f32 = output[measure_start..].iter().map(|x| x * x).sum();
        
        assert!(
            out_energy < in_energy * 0.5,
            "Echo not cancelled: in={:.4}, out={:.4}",
            in_energy,
            out_energy
        );
    }
}
