/// Acoustic Echo Cancellation module - NLMS Adaptive Filter Implementation
///
/// Uses the Normalized Least Mean Squares (NLMS) algorithm which automatically
/// learns the echo path (delay + gain + frequency response) without explicit
/// delay estimation. Works across different systems with varying latencies.

/// Filter length in samples - covers ~50ms at 48kHz stereo
/// This is a good balance between delay coverage and CPU usage
const FILTER_LEN: usize = 4800;

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
    /// Running sum of squares for power normalization
    power_sum: f32,
    /// Adaptation step size (mu) - controls convergence speed vs stability
    mu: f32,
    /// Regularization constant to prevent division by zero
    delta: f32,
    /// Filter length being used
    filter_len: usize,
    /// Frame counter for convergence tracking
    frame_count: usize,
    /// Whether filter has had time to start converging
    is_converging: bool,
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
    pub fn with_filter_len(filter_len: usize) -> Self {
        Self {
            filter: vec![0.0; filter_len],
            ref_buffer: vec![0.0; filter_len],
            buf_pos: 0,
            power_sum: 0.0,
            mu: 0.5,           // Adaptation rate
            delta: 1e-4,       // Regularization (larger for stability)
            filter_len,
            frame_count: 0,
            is_converging: false,
        }
    }

    pub fn reset(&mut self) {
        self.filter.fill(0.0);
        self.ref_buffer.fill(0.0);
        self.buf_pos = 0;
        self.power_sum = 0.0;
        self.frame_count = 0;
        self.is_converging = false;
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
        
        for i in 0..len {
            let ref_sample = reference[i];
            let in_sample = input[i];
            
            // Get the oldest sample that will be removed from power calculation
            let old_sample = self.ref_buffer[self.buf_pos];
            
            // Update running power sum (remove old, add new)
            self.power_sum = self.power_sum - old_sample * old_sample + ref_sample * ref_sample;
            // Ensure power_sum doesn't go negative due to floating point errors
            if self.power_sum < 0.0 {
                self.power_sum = 0.0;
            }
            
            // Store reference in circular buffer
            self.ref_buffer[self.buf_pos] = ref_sample;
            
            // Compute filter output (estimated echo)
            let echo_estimate = self.compute_filter_output();
            
            // Error = input - estimated echo (this is our output)
            let error = in_sample - echo_estimate;
            
            // Only adapt when we have sufficient reference energy
            // This prevents divergence during silence and double-talk
            let should_adapt = self.power_sum > self.delta * self.filter_len as f32;
            
            if should_adapt {
                // Mark that convergence has started
                if !self.is_converging {
                    self.is_converging = true;
                }
                
                // NLMS coefficient update with proper normalization
                // w(n+1) = w(n) + mu * e(n) * x(n) / (||x||^2 + delta)
                let norm = self.power_sum + self.delta;
                let step = self.mu * error / norm;
                
                // Update filter coefficients
                self.update_filter(step);
            }
            
            // Advance circular buffer position
            self.buf_pos = (self.buf_pos + 1) % self.filter_len;
            
            // Output the error signal (echo-cancelled)
            output.push(error.clamp(-1.0, 1.0));
        }
        
        self.frame_count += 1;
        output
    }

    /// Compute filter output using convolution
    #[inline]
    fn compute_filter_output(&self) -> f32 {
        let mut sum = 0.0f32;
        
        for i in 0..self.filter_len {
            // filter[0] corresponds to current sample, filter[1] to previous, etc.
            let buf_idx = (self.buf_pos + self.filter_len - i) % self.filter_len;
            sum += self.filter[i] * self.ref_buffer[buf_idx];
        }
        
        sum
    }

    /// Update filter coefficients
    #[inline]
    fn update_filter(&mut self, step: f32) {
        for i in 0..self.filter_len {
            let buf_idx = (self.buf_pos + self.filter_len - i) % self.filter_len;
            self.filter[i] += step * self.ref_buffer[buf_idx];
            
            // Soft limit coefficients to prevent explosion
            self.filter[i] = self.filter[i].clamp(-10.0, 10.0);
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
    fn test_passthrough_without_reference() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // With zero reference, output should equal input
        let reference = vec![0.0f32; 960];
        let input: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();
        
        let output = aec.process(&reference, &input);
        
        // Output should be very close to input
        let diff: f32 = output.iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).abs())
            .sum::<f32>() / output.len() as f32;
        
        assert!(diff < 0.01, "Output differs too much from input: {}", diff);
    }

    #[cfg(windows)]
    #[test]
    fn test_nlms_echo_cancellation() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Create reference signal
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

        // Process multiple frames to let NLMS converge
        for _ in 0..30 {
            let _ = aec.process(&reference, &input);
        }

        let output = aec.process(&reference, &input);

        // Measure energy reduction
        let measure_start = delay + 200;
        let in_energy: f32 = input[measure_start..].iter().map(|x| x * x).sum();
        let out_energy: f32 = output[measure_start..].iter().map(|x| x * x).sum();
        
        assert!(
            out_energy < in_energy,
            "Echo should be reduced: in={:.4}, out={:.4}",
            in_energy,
            out_energy
        );
    }
}
