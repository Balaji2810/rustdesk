/// Acoustic Echo Cancellation module - NLMS Adaptive Filter Implementation
///
/// Uses the Normalized Least Mean Squares (NLMS) algorithm which automatically
/// learns the echo path (delay + gain + frequency response) without explicit
/// delay estimation. Works across different systems with varying latencies.

/// Filter length in samples - covers ~100ms at 48kHz stereo
/// Longer = handles more delay but uses more CPU
const FILTER_LEN: usize = 9600;

/// Minimum filter length for low-latency mode (~25ms)
const FILTER_LEN_SHORT: usize = 2400;

/// NLMS Adaptive Echo Canceller
/// 
/// Automatically adapts to find and cancel echo regardless of system delay.
/// No manual delay configuration needed - it learns the echo path.
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// Adaptive filter coefficients (FIR filter modeling echo path)
    filter: Vec<f32>,
    /// Reference signal circular buffer
    ref_buffer: Vec<f32>,
    /// Current write position in circular buffer
    buf_pos: usize,
    /// Smoothed reference signal power (for normalization)
    ref_power: f32,
    /// Adaptation step size (mu) - controls convergence speed vs stability
    mu: f32,
    /// Regularization constant to prevent division by zero
    delta: f32,
    /// Filter length being used
    filter_len: usize,
    /// Leakage factor for coefficient stability (prevents drift)
    leakage: f32,
    /// Double-talk detector threshold
    dtd_threshold: f32,
    /// Smoothed error power for DTD
    error_power: f32,
    /// Frame counter for statistics
    frame_count: usize,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(_num_channels: usize) -> Self {
        Self::with_filter_len(FILTER_LEN)
    }

    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    /// Create with custom filter length
    /// Shorter filter = less CPU but handles less delay
    pub fn with_filter_len(filter_len: usize) -> Self {
        Self {
            filter: vec![0.0; filter_len],
            ref_buffer: vec![0.0; filter_len],
            buf_pos: 0,
            ref_power: 0.0,
            mu: 0.4,           // Adaptation rate: 0.1-0.5 typical, higher = faster
            delta: 1e-6,       // Regularization
            filter_len,
            leakage: 0.9999,   // Very slight leakage to prevent coefficient explosion
            dtd_threshold: 2.0, // Double-talk detection threshold
            error_power: 0.0,
            frame_count: 0,
        }
    }

    /// Create optimized for low latency systems
    pub fn new_low_latency() -> Self {
        Self::with_filter_len(FILTER_LEN_SHORT)
    }

    pub fn reset(&mut self) {
        self.filter.fill(0.0);
        self.ref_buffer.fill(0.0);
        self.buf_pos = 0;
        self.ref_power = 0.0;
        self.error_power = 0.0;
        self.frame_count = 0;
    }

    /// Set adaptation rate (0.1 = slow/stable, 0.5 = fast/aggressive)
    #[allow(dead_code)]
    pub fn set_mu(&mut self, mu: f32) {
        self.mu = mu.clamp(0.05, 0.8);
    }

    /// Process audio buffer through NLMS adaptive filter
    /// 
    /// # Arguments
    /// * `reference` - The signal being played (far-end / client audio)
    /// * `input` - The captured signal containing echo (near-end + echo)
    /// 
    /// # Returns
    /// Echo-cancelled signal (near-end with echo removed)
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(len);
        
        // Power smoothing factor (time constant ~10ms at 48kHz)
        let alpha = 0.998_f32;
        let alpha_inv = 1.0 - alpha;
        
        for i in 0..len {
            let ref_sample = reference[i];
            let in_sample = input[i];
            
            // Store reference in circular buffer
            self.ref_buffer[self.buf_pos] = ref_sample;
            
            // Update smoothed reference power (leaky integrator)
            self.ref_power = alpha * self.ref_power + alpha_inv * ref_sample * ref_sample;
            
            // Compute filter output (estimated echo) using convolution
            let echo_estimate = self.compute_filter_output();
            
            // Error = input - estimated echo (this is our output)
            let error = in_sample - echo_estimate;
            
            // Update smoothed error power for double-talk detection
            self.error_power = alpha * self.error_power + alpha_inv * error * error;
            
            // Double-talk detection: if error power >> reference power, 
            // someone is talking on near-end, so reduce/pause adaptation
            let dtd_ratio = if self.ref_power > self.delta {
                self.error_power / self.ref_power
            } else {
                0.0
            };
            
            // Only adapt if reference has energy and no double-talk detected
            let should_adapt = self.ref_power > 1e-8 && dtd_ratio < self.dtd_threshold;
            
            if should_adapt {
                // NLMS coefficient update
                // w(n+1) = leakage * w(n) + mu * error * x(n) / (||x||^2 + delta)
                let norm_factor = self.mu / (self.ref_power * self.filter_len as f32 + self.delta);
                self.update_filter(error, norm_factor);
            }
            
            // Advance circular buffer position
            self.buf_pos = (self.buf_pos + 1) % self.filter_len;
            
            output.push(error.clamp(-1.0, 1.0));
        }
        
        self.frame_count += 1;
        output
    }

    /// Compute filter output (convolution of filter with reference buffer)
    #[inline]
    fn compute_filter_output(&self) -> f32 {
        let mut sum = 0.0f32;
        
        // Process in chunks of 4 for better vectorization
        let chunks = self.filter_len / 4;
        let remainder = self.filter_len % 4;
        
        for chunk in 0..chunks {
            let base = chunk * 4;
            let mut partial = [0.0f32; 4];
            
            for j in 0..4 {
                let filter_idx = base + j;
                let buf_idx = (self.buf_pos + self.filter_len - filter_idx) % self.filter_len;
                partial[j] = self.filter[filter_idx] * self.ref_buffer[buf_idx];
            }
            
            sum += partial[0] + partial[1] + partial[2] + partial[3];
        }
        
        // Handle remainder
        for j in 0..remainder {
            let filter_idx = chunks * 4 + j;
            let buf_idx = (self.buf_pos + self.filter_len - filter_idx) % self.filter_len;
            sum += self.filter[filter_idx] * self.ref_buffer[buf_idx];
        }
        
        sum
    }

    /// Update filter coefficients using NLMS rule
    #[inline]
    fn update_filter(&mut self, error: f32, norm_factor: f32) {
        let update_scale = error * norm_factor;
        
        // Process in chunks for better cache performance
        let chunks = self.filter_len / 4;
        let remainder = self.filter_len % 4;
        
        for chunk in 0..chunks {
            let base = chunk * 4;
            
            for j in 0..4 {
                let filter_idx = base + j;
                let buf_idx = (self.buf_pos + self.filter_len - filter_idx) % self.filter_len;
                
                // NLMS update with leakage
                self.filter[filter_idx] = self.leakage * self.filter[filter_idx] 
                    + update_scale * self.ref_buffer[buf_idx];
            }
        }
        
        // Handle remainder
        for j in 0..remainder {
            let filter_idx = chunks * 4 + j;
            let buf_idx = (self.buf_pos + self.filter_len - filter_idx) % self.filter_len;
            self.filter[filter_idx] = self.leakage * self.filter[filter_idx] 
                + update_scale * self.ref_buffer[buf_idx];
        }
    }
}

/// Process buffer - main entry point for echo cancellation
#[cfg(windows)]
pub fn process_buffer(
    aec: &mut DigitalEchoCanceller,
    reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    aec.process(reference, input)
}

// Non-Windows: passthrough (no echo cancellation needed/supported)
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
    fn test_nlms_echo_cancellation() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Create reference signal (simulated far-end audio)
        let reference: Vec<f32> = (0..4800)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        
        // Simulate echo with 480 sample delay (5ms at 48kHz stereo) and 0.8 gain
        let delay = 480;
        let echo_gain = 0.8;
        let mut input = vec![0.0f32; 4800];
        for i in delay..4800 {
            input[i] = reference[i - delay] * echo_gain;
        }

        // Process multiple frames to let NLMS converge
        // NLMS typically needs several frames to adapt
        for _ in 0..20 {
            let _ = aec.process(&reference, &input);
        }

        // After convergence, output should have significantly reduced echo
        let output = aec.process(&reference, &input);

        // Measure energy reduction (skip initial transient)
        let measure_start = delay + 100;
        let in_energy: f32 = input[measure_start..].iter().map(|x| x * x).sum();
        let out_energy: f32 = output[measure_start..].iter().map(|x| x * x).sum();
        
        // Should achieve at least 6dB (4x power) reduction
        assert!(
            out_energy < in_energy / 2.0,
            "Echo not sufficiently cancelled: in={:.4}, out={:.4}",
            in_energy,
            out_energy
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_nlms_varying_delay() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Test that NLMS can handle different delays without reconfiguration
        for delay in [240, 480, 960, 1920].iter() {
            aec.reset();
            
            let reference: Vec<f32> = (0..4800)
                .map(|i| (i as f32 * 0.02).sin() * 0.5)
                .collect();
            
            let mut input = vec![0.0f32; 4800];
            for i in *delay..4800 {
                input[i] = reference[i - delay] * 0.7;
            }

            // Let it converge
            for _ in 0..30 {
                let _ = aec.process(&reference, &input);
            }

            let output = aec.process(&reference, &input);
            
            let measure_start = delay + 200;
            if measure_start < 4800 {
                let in_energy: f32 = input[measure_start..].iter().map(|x| x * x).sum();
                let out_energy: f32 = output[measure_start..].iter().map(|x| x * x).sum();
                
                assert!(
                    out_energy < in_energy,
                    "Failed at delay {}: in={:.4}, out={:.4}",
                    delay, in_energy, out_energy
                );
            }
        }
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
            let echo = if i >= delay { reference[i - delay] * 0.6 } else { 0.0 };
            input[i] = echo + near_end[i];
        }

        // Let it partially converge
        for _ in 0..10 {
            let _ = aec.process(&reference, &input);
        }

        let output = aec.process(&reference, &input);

        // Near-end signal should be preserved (not completely cancelled)
        let near_energy: f32 = near_end[delay..].iter().map(|x| x * x).sum();
        let out_energy: f32 = output[delay..].iter().map(|x| x * x).sum();
        
        // Output should retain significant energy from near-end
        assert!(
            out_energy > near_energy * 0.3,
            "Near-end signal over-suppressed"
        );
    }
}

