/// Digital Echo Cancellation module - Pure Rust implementation
///
/// Optimized for **digital** echo paths (loopback) where echo is a
/// delayed copy of the reference signal. Uses simple delay-and-subtract
/// which is the fastest approach for digital echo with minimal latency.
use std::collections::VecDeque;

/// Maximum delay to compensate (samples) - ~100ms at 48kHz stereo
const MAX_DELAY: usize = 9600;

/// Correlation window size (samples) - ~5ms at 48kHz stereo
const CORR_WINDOW: usize = 480;

/// Digital Echo Canceller - optimized for loopback echo
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// Reference signal ring buffer
    ref_buffer: VecDeque<f32>,
    /// Estimated delay in samples
    delay: usize,
    /// Estimated gain
    gain: f32,
    /// Frame counter for periodic delay estimation
    frame_count: usize,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(_num_channels: usize) -> Self {
        Self {
            ref_buffer: VecDeque::with_capacity(MAX_DELAY + CORR_WINDOW),
            delay: 0,//960, // Initial guess: 10ms at 48kHz stereo
            gain: 1.0,
            frame_count: 0,
        }
    }

    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    pub fn reset(&mut self) {
        self.ref_buffer.clear();
        self.delay = 0;//960;
        self.gain = 1.0;
        self.frame_count = 0;
    }

    /// Process audio - subtract delayed reference from input
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // Add reference to buffer
        for &s in reference.iter().take(len) {
            if self.ref_buffer.len() >= MAX_DELAY + CORR_WINDOW {
                self.ref_buffer.pop_front();
            }
            self.ref_buffer.push_back(s);
        }

        // Not enough history yet
        if self.ref_buffer.len() < self.delay + len {
            return input[..len].to_vec();
        }

        // DISABLED: Delay estimation is causing issues - use fixed delay instead
        // // Update delay estimate every ~50 frames (~500ms)
        // self.frame_count += 1;
        // if self.frame_count >= 50 && len >= CORR_WINDOW {
        //     self.frame_count = 0;
        //     self.estimate_delay(input);
        // }

        // Cancel echo: output = input - gain * delayed_reference
        let buf_len = self.ref_buffer.len();
        let ref_start = buf_len.saturating_sub(self.delay + len);

        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let ref_sample = self.ref_buffer.get(ref_start + i).copied().unwrap_or(0.0);
            let cancelled = input[i] - ref_sample * self.gain;
            output.push(cancelled.clamp(-1.0, 1.0));
        }

        output
    }

    /// Estimate delay using normalized cross-correlation
    fn estimate_delay(&mut self, input: &[f32]) {
        let buf_len = self.ref_buffer.len();
        if buf_len < CORR_WINDOW + MAX_DELAY {
            return;
        }

        let window = CORR_WINDOW.min(input.len());
        let mut best_corr = 0.0f32;
        let mut best_delay = self.delay;
        let mut best_gain = self.gain;

        // Coarse search: step by 16 samples
        for delay in (96..MAX_DELAY).step_by(16) {
            let (corr, gain) = self.correlation_at_delay(input, delay, window);
            if corr > best_corr {
                best_corr = corr;
                best_delay = delay;
                best_gain = gain;
            }
        }

        // Fine search around best delay
        if best_corr > 0.3 {
            let start = best_delay.saturating_sub(32);
            let end = (best_delay + 32).min(MAX_DELAY);
            for delay in (start..end).step_by(2) {
                let (corr, gain) = self.correlation_at_delay(input, delay, window);
                if corr > best_corr {
                    best_corr = corr;
                    best_delay = delay;
                    best_gain = gain;
                }
            }
        }

        // Update with smoothing if correlation is strong
        if best_corr > 0.5 {
            self.delay = (self.delay * 7 + best_delay) / 8; // Smooth delay
            self.gain = self.gain * 0.9 + best_gain * 0.1; // Smooth gain
            self.gain = self.gain.clamp(0.5, 1.5);
        }
    }

    /// Compute normalized correlation and gain at a specific delay
    fn correlation_at_delay(&self, input: &[f32], delay: usize, window: usize) -> (f32, f32) {
        let buf_len = self.ref_buffer.len();
        let ref_start = buf_len.saturating_sub(delay + window);

        let mut sum_xy = 0.0f32;
        let mut sum_x2 = 0.0f32;
        let mut sum_y2 = 0.0f32;

        for i in 0..window {
            let x = input[i];
            let y = self.ref_buffer.get(ref_start + i).copied().unwrap_or(0.0);
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        let corr = if denom > 1e-6 { sum_xy / denom } else { 0.0 };
        let gain = if sum_y2 > 1e-6 { sum_xy / sum_y2 } else { 1.0 };

        (corr.abs(), gain.clamp(0.5, 1.5))
    }
}

/// Process buffer - direct passthrough to process()
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
    fn test_echo_cancellation() {
        let mut dec = DigitalEchoCanceller::new_stereo();

        // Create reference and echoed input
        let reference: Vec<f32> = (0..1920).map(|i| (i as f32 * 0.01).sin()).collect();
        let delay = 480;
        let mut input = vec![0.0f32; 1920];
        for i in delay..1920 {
            input[i] = reference[i - delay] * 0.9;
        }

        // Process multiple times to let it converge
        for _ in 0..5 {
            let _ = dec.process(&reference, &input);
        }

        let output = dec.process(&reference, &input);

        // Output should have reduced energy
        let in_energy: f32 = input.iter().map(|x| x * x).sum();
        let out_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(out_energy <= in_energy);
    }
}
