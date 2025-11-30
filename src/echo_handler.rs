/// Acoustic Echo Cancellation module using WebRTC AEC3
///
/// Uses the aec3 crate - a pure Rust port of WebRTC's AEC3 algorithm.
/// This is the industry-standard echo cancellation used by Chrome, etc.

/// Frame size for 48kHz (10ms frames)
const FRAME_SIZE_48K: usize = 480;

#[cfg(windows)]
use aec3::voip::VoipAec3;

/// WebRTC AEC3 Echo Canceller wrapper
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    /// The AEC3 processor
    aec: VoipAec3,
    /// Number of channels (1 = mono, 2 = stereo)
    num_channels: usize,
    /// Sample rate
    sample_rate: u32,
    /// Frame size in samples per channel
    frame_size: usize,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(num_channels: usize) -> Self {
        Self::with_sample_rate(48000, num_channels)
    }

    pub fn new_stereo() -> Self {
        Self::new(2)
    }

    /// Create with specific sample rate and channel count
    pub fn with_sample_rate(sample_rate: u32, num_channels: usize) -> Self {
        let frame_size = match sample_rate {
            16000 => 160,
            32000 => 320,
            48000 => 480,
            _ => FRAME_SIZE_48K, // Default to 48kHz frame size
        };

        let aec = VoipAec3::builder(sample_rate as i32, num_channels as i32, num_channels as i32)
            .enable_high_pass(true)
            .initial_delay_ms(0)
            .build()
            .expect("Failed to build AEC3");

        Self {
            aec,
            num_channels,
            sample_rate,
            frame_size,
        }
    }

    pub fn reset(&mut self) {
        // Recreate the AEC processor
        self.aec = VoipAec3::builder(
            self.sample_rate as i32,
            self.num_channels as i32,
            self.num_channels as i32,
        )
        .enable_high_pass(true)
        .initial_delay_ms(0)
        .build()
        .expect("Failed to rebuild AEC3");
    }

    /// Process audio buffer through AEC3
    ///
    /// # Arguments
    /// * `reference` - The signal being played (far-end / client audio) - interleaved samples
    /// * `input` - The captured signal containing echo (near-end + echo) - interleaved samples
    ///
    /// # Returns
    /// Echo-cancelled signal (near-end with echo removed)
    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        let len = reference.len().min(input.len());
        if len == 0 {
            return Vec::new();
        }

        // Frame size in total samples (samples per channel * channels)
        let frame_samples = self.frame_size * self.num_channels;
        let mut output = Vec::with_capacity(len);

        // Process in 10ms frames as required by AEC3
        let mut pos = 0;
        while pos + frame_samples <= len {
            let ref_frame = &reference[pos..pos + frame_samples];
            let cap_frame = &input[pos..pos + frame_samples];

            // Feed reference (render) frame to AEC - this is what's being played on speakers
            self.aec.handle_render_frame(ref_frame);

            // Process capture frame and get echo-cancelled output
            let mut out_frame = vec![0.0f32; frame_samples];
            if let Err(e) = self.aec.process_capture_frame(cap_frame, false, &mut out_frame) {
                log::warn!("AEC3 process error: {:?}", e);
                // On error, pass through the input
                output.extend_from_slice(cap_frame);
            } else {
                output.extend_from_slice(&out_frame);
            }

            pos += frame_samples;
        }

        // Handle remaining samples (less than one frame)
        // Just pass them through unchanged
        if pos < len {
            output.extend_from_slice(&input[pos..len]);
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
    fn test_aec3_creation() {
        let aec = DigitalEchoCanceller::new_stereo();
        assert_eq!(aec.num_channels, 2);
        assert_eq!(aec.sample_rate, 48000);
        assert_eq!(aec.frame_size, 480);
    }

    #[cfg(windows)]
    #[test]
    fn test_aec3_process_empty() {
        let mut aec = DigitalEchoCanceller::new_stereo();
        let result = aec.process(&[], &[]);
        assert!(result.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn test_aec3_process_frames() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Create test signals (10ms frame at 48kHz stereo = 960 samples)
        let frame_size = 480 * 2; // stereo
        let reference: Vec<f32> = (0..frame_size)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        // Simulate echo
        let input: Vec<f32> = reference.iter().map(|x| x * 0.7).collect();

        let output = aec.process(&reference, &input);
        assert_eq!(output.len(), frame_size);
    }

    #[cfg(windows)]
    #[test]
    fn test_aec3_reset() {
        let mut aec = DigitalEchoCanceller::new_stereo();

        // Process some data
        let frame_size = 480 * 2;
        let reference: Vec<f32> = vec![0.1; frame_size];
        let input: Vec<f32> = vec![0.1; frame_size];
        let _ = aec.process(&reference, &input);

        // Reset and verify it still works
        aec.reset();
        let output = aec.process(&reference, &input);
        assert_eq!(output.len(), frame_size);
    }
}
