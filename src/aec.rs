/// Acoustic Echo Cancellation module using SpeexDSP
///
/// This module wraps the `aec` crate which provides SpeexDSP-based
/// echo cancellation. SpeexDSP is a battle-tested DSP library specifically
/// designed for VoIP applications.

/// Default sample rate for audio processing
const SAMPLE_RATE: u32 = 48000;

/// Frame size for processing (10ms at 48kHz)
/// SpeexDSP works best with 10-20ms frames
const FRAME_SIZE: usize = 480;

/// Filter length in milliseconds (tail length for echo)
/// 100ms should handle most digital loopback scenarios
const FILTER_LENGTH_MS: u32 = 100;

/// Echo Canceller wrapper using SpeexDSP
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// The SpeexDSP AEC instance
    aec: aec::Aec,
    /// Sample rate
    sample_rate: u32,
    /// Frame size for processing
    frame_size: usize,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(_num_channels: usize) -> Self {
        Self::with_params(SAMPLE_RATE, FRAME_SIZE, FILTER_LENGTH_MS)
    }

    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    /// Create with custom parameters
    pub fn with_params(sample_rate: u32, frame_size: usize, filter_length_ms: u32) -> Self {
        // Calculate filter length in samples
        let filter_length = (sample_rate * filter_length_ms / 1000) as usize;
        
        let aec = aec::Aec::new(
            1,             // channels (process as mono, we'll handle stereo ourselves)
            sample_rate,
            frame_size,
            filter_length,
        ).expect("Failed to create SpeexDSP AEC");

        Self {
            aec,
            sample_rate,
            frame_size,
        }
    }

    pub fn reset(&mut self) {
        // Recreate the AEC instance to reset it
        let filter_length = (self.sample_rate * FILTER_LENGTH_MS / 1000) as usize;
        self.aec = aec::Aec::new(
            1,
            self.sample_rate,
            self.frame_size,
            filter_length,
        ).expect("Failed to reset SpeexDSP AEC");
    }

    /// Process audio through SpeexDSP echo canceller
    /// 
    /// # Arguments
    /// * `reference` - The far-end signal (client audio being played on speakers)
    /// * `input` - The near-end signal (captured audio containing echo)
    /// 
    /// # Returns
    /// Echo-cancelled audio
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // For stereo audio, process as interleaved stereo
        // SpeexDSP expects mono, so we'll process left and right channels together
        // by averaging them or processing interleaved
        
        let mut output = Vec::with_capacity(len);
        
        // Process in frame-sized chunks
        let mut pos = 0;
        while pos < len {
            let chunk_size = self.frame_size.min(len - pos);
            let end = pos + chunk_size;
            
            // Get reference and input chunks
            let ref_chunk: Vec<i16> = reference[pos..end]
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            
            let in_chunk: Vec<i16> = input[pos..end]
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            
            // Process through SpeexDSP AEC
            // First, feed the reference (playback) signal
            self.aec.playback(&ref_chunk);
            
            // Then process the capture signal to remove echo
            let mut out_chunk = vec![0i16; chunk_size];
            self.aec.capture(&in_chunk, &mut out_chunk);
            
            // Convert back to f32 and append to output
            for &s in &out_chunk {
                output.push(s as f32 / 32767.0);
            }
            
            pos = end;
        }
        
        output
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

// Non-Windows: passthrough (no echo cancellation)
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
    #[allow(unused_imports)]
    use super::*;

    #[cfg(windows)]
    #[test]
    fn test_aec_creation() {
        let aec = DigitalEchoCanceller::new_stereo();
        assert_eq!(aec.sample_rate, SAMPLE_RATE);
        assert_eq!(aec.frame_size, FRAME_SIZE);
    }

    #[cfg(windows)]
    #[test]
    fn test_aec_process() {
        let mut aec = DigitalEchoCanceller::new_stereo();
        
        // Create test signals
        let reference: Vec<f32> = (0..960)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let input: Vec<f32> = (0..960)
            .map(|i| (i as f32 * 0.01).sin() * 0.4)  // Echo of reference
            .collect();
        
        let output = aec.process(&reference, &input);
        
        // Output should have same length as input
        assert_eq!(output.len(), input.len());
        
        // Output values should be in valid range
        for &s in &output {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }
}
