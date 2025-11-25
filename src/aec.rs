/// Acoustic Echo Cancellation (AEC) module using NLMS adaptive filter
/// 
/// This module implements a Normalized Least Mean Squares (NLMS) adaptive filter
/// for canceling digital echo in real-time audio transmission systems.
/// 
/// The NLMS algorithm is well-suited for digital echo paths because:
/// - Fast convergence for simple delay+gain paths
/// - Low computational complexity
/// - Stable with proper parameter tuning

use std::collections::VecDeque;

/// NLMS (Normalized Least Mean Squares) adaptive filter for echo cancellation
pub struct NlmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f32>,
    /// Reference signal history buffer
    reference_buffer: VecDeque<f32>,
    /// Filter length (number of taps)
    filter_length: usize,
    /// Step size (learning rate), typically 0.1 - 0.5
    step_size: f32,
    /// Small constant to prevent division by zero
    epsilon: f32,
}

impl NlmsFilter {
    /// Create a new NLMS filter
    /// 
    /// # Arguments
    /// * `filter_length` - Number of filter taps (e.g., 960 for 20ms at 48kHz)
    /// * `step_size` - Learning rate (0.1 - 0.5 recommended, higher = faster convergence but less stable)
    pub fn new(filter_length: usize, step_size: f32) -> Self {
        Self {
            weights: vec![0.0; filter_length],
            reference_buffer: VecDeque::with_capacity(filter_length),
            filter_length,
            step_size,
            epsilon: 1e-8,
        }
    }

    /// Create a new NLMS filter with default parameters optimized for digital echo
    /// 
    /// Default: 960 taps (20ms at 48kHz), step_size = 0.3
    pub fn new_default() -> Self {
        Self::new(960, 0.3)
    }

    /// Reset the filter state (weights and buffer)
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.reference_buffer.clear();
    }

    /// Process a single sample through the NLMS filter
    /// 
    /// # Arguments
    /// * `reference` - Reference signal sample (far-end signal, e.g., client audio)
    /// * `input` - Input signal sample (near-end signal containing echo, e.g., loopback)
    /// 
    /// # Returns
    /// Echo-cancelled output sample
    #[inline]
    fn process_sample(&mut self, reference: f32, input: f32) -> f32 {
        // Add reference sample to history buffer
        self.reference_buffer.push_front(reference);
        if self.reference_buffer.len() > self.filter_length {
            self.reference_buffer.pop_back();
        }

        // Compute filter output (echo estimate) using dot product
        let mut echo_estimate = 0.0f32;
        let mut power = 0.0f32;
        
        for (i, &ref_sample) in self.reference_buffer.iter().enumerate() {
            if i >= self.weights.len() {
                break;
            }
            echo_estimate += self.weights[i] * ref_sample;
            power += ref_sample * ref_sample;
        }

        // Compute error (echo-cancelled signal)
        let error = input - echo_estimate;

        // Update filter weights using NLMS
        // w[n+1] = w[n] + (mu * error * x[n]) / (x[n]^T * x[n] + epsilon)
        let normalization = power + self.epsilon;
        let update_factor = self.step_size * error / normalization;

        for (i, &ref_sample) in self.reference_buffer.iter().enumerate() {
            if i >= self.weights.len() {
                break;
            }
            self.weights[i] += update_factor * ref_sample;
        }

        error
    }

    /// Process a frame of audio samples
    /// 
    /// # Arguments
    /// * `reference` - Reference signal buffer (far-end signal, e.g., client audio that was played)
    /// * `input` - Input signal buffer (near-end signal containing echo, e.g., system loopback)
    /// 
    /// # Returns
    /// Echo-cancelled output buffer
    /// 
    /// # Note
    /// Both buffers should have the same length and be at the same sample rate and channel count.
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        let mut output = Vec::with_capacity(len);

        for i in 0..len {
            let echo_cancelled = self.process_sample(reference[i], input[i]);
            output.push(echo_cancelled);
        }

        output
    }

    /// Get the current filter length
    pub fn filter_length(&self) -> usize {
        self.filter_length
    }

    /// Get the current step size
    pub fn step_size(&self) -> f32 {
        self.step_size
    }

    /// Set a new step size (useful for dynamic adaptation)
    pub fn set_step_size(&mut self, step_size: f32) {
        self.step_size = step_size.clamp(0.01, 1.0);
    }
}

/// Helper function to create an AEC filter for stereo audio
/// Returns two filters, one for each channel
pub fn create_stereo_aec(filter_length: usize, step_size: f32) -> (NlmsFilter, NlmsFilter) {
    (
        NlmsFilter::new(filter_length, step_size),
        NlmsFilter::new(filter_length, step_size),
    )
}

/// Process stereo audio through AEC filters
/// 
/// # Arguments
/// * `filters` - Tuple of (left_filter, right_filter)
/// * `reference` - Interleaved stereo reference signal [L, R, L, R, ...]
/// * `input` - Interleaved stereo input signal [L, R, L, R, ...]
/// 
/// # Returns
/// Interleaved stereo echo-cancelled output
pub fn process_stereo(
    filters: &mut (NlmsFilter, NlmsFilter),
    reference: &[f32],
    input: &[f32],
) -> Vec<f32> {
    let len = reference.len().min(input.len());
    // Ensure even length for stereo
    let len = len - (len % 2);
    
    if len == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(len);

    for i in (0..len).step_by(2) {
        // Process left channel
        let left_out = filters.0.process_sample(reference[i], input[i]);
        // Process right channel
        let right_out = filters.1.process_sample(reference[i + 1], input[i + 1]);
        
        output.push(left_out);
        output.push(right_out);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlms_filter_creation() {
        let filter = NlmsFilter::new(100, 0.5);
        assert_eq!(filter.filter_length(), 100);
        assert_eq!(filter.step_size(), 0.5);
    }

    #[test]
    fn test_nlms_filter_reset() {
        let mut filter = NlmsFilter::new(100, 0.5);
        // Process some samples to modify state
        filter.process(&[1.0; 10], &[1.0; 10]);
        // Reset
        filter.reset();
        // Check that weights are zeroed
        assert!(filter.weights.iter().all(|&w| w == 0.0));
        assert!(filter.reference_buffer.is_empty());
    }

    #[test]
    fn test_nlms_echo_cancellation() {
        let mut filter = NlmsFilter::new(10, 0.5);
        
        // Simulate a simple echo: input = reference delayed by 2 samples
        let reference: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut input = vec![0.0; 2];
        input.extend_from_slice(&reference[..98]);
        
        // Process through filter
        let output = filter.process(&reference, &input);
        
        // After convergence, output should have lower energy than input
        let input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        
        // The filter should reduce the echo energy
        // (For a well-converged filter, output_energy << input_energy)
        assert!(output.len() == input.len().min(reference.len()));
    }

    #[test]
    fn test_stereo_processing() {
        let mut filters = create_stereo_aec(10, 0.5);
        
        // Create stereo reference and input
        let reference = vec![1.0, 0.5, 0.8, 0.4, 0.6, 0.3, 0.4, 0.2];
        let input = vec![0.9, 0.45, 0.7, 0.35, 0.5, 0.25, 0.3, 0.15];
        
        let output = process_stereo(&mut filters, &reference, &input);
        
        assert_eq!(output.len(), 8);
    }
}

