/// Acoustic Echo Cancellation (AEC) module using WebRTC's audio processing
///
/// This module wraps WebRTC's highly optimized AEC3 algorithm for real-time
/// echo cancellation with minimal latency.

#[cfg(windows)]
use std::sync::Arc;

#[cfg(windows)]
use webrtc_audio_processing::{
    Config, EchoCancellation, EchoCancellationSuppressionLevel, InitializationConfig,
    NoiseSuppression, NoiseSuppressionLevel, Processor, NUM_SAMPLES_PER_FRAME,
};

/// WebRTC Audio Processor for echo cancellation
///
/// This wraps WebRTC's audio processing module which includes:
/// - AEC3 (Acoustic Echo Cancellation)
/// - Noise suppression
/// - Optimized for real-time with minimal latency
#[cfg(windows)]
pub struct WebRtcAec {
    processor: Processor,
    /// Number of channels
    num_channels: usize,
    /// Frame size (samples per channel per frame)
    frame_size: usize,
}

#[cfg(windows)]
impl WebRtcAec {
    /// Create a new WebRTC AEC processor
    ///
    /// # Arguments
    /// * `num_channels` - Number of audio channels (1 for mono, 2 for stereo)
    pub fn new(num_channels: usize) -> Result<Self, String> {
        let init_config = InitializationConfig {
            num_capture_channels: num_channels,
            num_render_channels: num_channels,
            ..Default::default()
        };

        let processor = Processor::new(&init_config)
            .map_err(|e| format!("Failed to create WebRTC processor: {:?}", e))?;

        let mut aec = Self {
            processor,
            num_channels,
            frame_size: NUM_SAMPLES_PER_FRAME,
        };

        // Configure echo cancellation
        let mut config = Config::default();
        config.echo_cancellation = EchoCancellation {
            suppression_level: EchoCancellationSuppressionLevel::High,
            enable_extended_filter: true,
            enable_delay_agnostic: true,
            stream_delay_ms: None,
        };
        // Also enable noise suppression for cleaner output
        config.noise_suppression = NoiseSuppression {
            suppression_level: NoiseSuppressionLevel::High,
        };

        aec.processor.set_config(config);

        Ok(aec)
    }

    /// Create a new WebRTC AEC processor for stereo audio
    pub fn new_stereo() -> Result<Self, String> {
        Self::new(2)
    }

    /// Reset the AEC state
    pub fn reset(&mut self) {
        // Re-create processor to reset state
        let init_config = InitializationConfig {
            num_capture_channels: self.num_channels,
            num_render_channels: self.num_channels,
            ..Default::default()
        };

        if let Ok(new_processor) = Processor::new(&init_config) {
            self.processor = new_processor;

            let mut config = Config::default();
            config.echo_cancellation = EchoCancellation {
                suppression_level: EchoCancellationSuppressionLevel::High,
                enable_extended_filter: true,
                enable_delay_agnostic: true,
                stream_delay_ms: None,
            };
            config.noise_suppression = NoiseSuppression {
                suppression_level: NoiseSuppressionLevel::High,
            };
            self.processor.set_config(config);
        }
    }

    /// Process render frame (far-end signal - audio being played to speakers)
    ///
    /// Call this with the audio data that will be played through speakers.
    /// This tells the AEC what audio to remove from the capture signal.
    ///
    /// # Arguments
    /// * `frame` - Deinterleaved audio channels, each with NUM_SAMPLES_PER_FRAME samples
    pub fn process_render_frame(&mut self, frame: &[Vec<i16>]) -> Result<(), String> {
        self.processor
            .process_render_frame(frame)
            .map_err(|e| format!("Failed to process render frame: {:?}", e))
    }

    /// Process capture frame (near-end signal - audio captured from microphone/loopback)
    ///
    /// Call this with the audio data captured from microphone/loopback.
    /// The AEC will remove the echo from this signal.
    ///
    /// # Arguments
    /// * `frame` - Deinterleaved audio channels, each with NUM_SAMPLES_PER_FRAME samples
    ///
    /// # Returns
    /// The frame is modified in place with echo cancelled
    pub fn process_capture_frame(&mut self, frame: &mut [Vec<i16>]) -> Result<(), String> {
        self.processor
            .process_capture_frame(frame)
            .map_err(|e| format!("Failed to process capture frame: {:?}", e))
    }

    /// Get the frame size required by WebRTC (samples per channel)
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

/// Helper to convert interleaved f32 to deinterleaved i16
#[cfg(windows)]
fn f32_interleaved_to_i16_deinterleaved(interleaved: &[f32], num_channels: usize) -> Vec<Vec<i16>> {
    let samples_per_channel = interleaved.len() / num_channels;
    let mut deinterleaved = vec![Vec::with_capacity(samples_per_channel); num_channels];

    for (i, &sample) in interleaved.iter().enumerate() {
        let channel = i % num_channels;
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        deinterleaved[channel].push(sample_i16);
    }

    deinterleaved
}

/// Helper to convert deinterleaved i16 to interleaved f32
#[cfg(windows)]
fn i16_deinterleaved_to_f32_interleaved(deinterleaved: &[Vec<i16>]) -> Vec<f32> {
    if deinterleaved.is_empty() || deinterleaved[0].is_empty() {
        return Vec::new();
    }

    let num_channels = deinterleaved.len();
    let samples_per_channel = deinterleaved[0].len();
    let mut interleaved = Vec::with_capacity(samples_per_channel * num_channels);

    for i in 0..samples_per_channel {
        for channel in deinterleaved {
            let sample_f32 = channel[i] as f32 / 32768.0;
            interleaved.push(sample_f32);
        }
    }

    interleaved
}

/// Process audio buffer through WebRTC AEC
///
/// Handles arbitrary buffer sizes by splitting into frames as required by WebRTC.
///
/// # Arguments
/// * `aec` - The WebRTC AEC processor
/// * `reference` - Reference signal (far-end, what was played) - interleaved f32
/// * `input` - Input signal (near-end, what was captured) - interleaved f32
///
/// # Returns
/// Echo-cancelled output buffer (interleaved f32)
#[cfg(windows)]
pub fn process_buffer(aec: &mut WebRtcAec, reference: &[f32], input: &[f32]) -> Vec<f32> {
    let num_channels = aec.num_channels();
    let frame_samples = NUM_SAMPLES_PER_FRAME; // samples per channel per frame
    let frame_size = frame_samples * num_channels; // total samples per frame

    let len = reference.len().min(input.len());
    if len < frame_size {
        // Not enough samples for a full frame, return input unchanged
        return input[..len].to_vec();
    }

    let mut output = Vec::with_capacity(len);
    let mut offset = 0;

    while offset + frame_size <= len {
        // Extract frame from reference
        let ref_frame = &reference[offset..offset + frame_size];
        let ref_deinterleaved = f32_interleaved_to_i16_deinterleaved(ref_frame, num_channels);

        // Process render frame (tell AEC what's being played)
        if let Err(e) = aec.process_render_frame(&ref_deinterleaved) {
            log::warn!("AEC render error: {}", e);
        }

        // Extract frame from input
        let input_frame = &input[offset..offset + frame_size];
        let mut input_deinterleaved =
            f32_interleaved_to_i16_deinterleaved(input_frame, num_channels);

        // Process capture frame (remove echo)
        if let Err(e) = aec.process_capture_frame(&mut input_deinterleaved) {
            log::warn!("AEC capture error: {}", e);
            // On error, just copy input
            output.extend_from_slice(input_frame);
        } else {
            // Convert back to interleaved f32
            let processed = i16_deinterleaved_to_f32_interleaved(&input_deinterleaved);
            output.extend_from_slice(&processed);
        }

        offset += frame_size;
    }

    // Handle remaining samples (less than a full frame) - pass through
    if offset < len {
        output.extend_from_slice(&input[offset..len]);
    }

    output
}

// Wrapper struct for compatibility with existing code
#[cfg(windows)]
pub struct DigitalEchoCanceller {
    aec: Option<WebRtcAec>,
}

#[cfg(windows)]
impl DigitalEchoCanceller {
    pub fn new(num_channels: usize) -> Self {
        Self {
            aec: WebRtcAec::new(num_channels).ok(),
        }
    }

    pub fn new_stereo() -> Self {
        Self {
            aec: WebRtcAec::new_stereo().ok(),
        }
    }

    pub fn reset(&mut self) {
        if let Some(aec) = &mut self.aec {
            aec.reset();
        }
    }

    pub fn process(&mut self, reference: &[f32], input: &[f32]) -> Vec<f32> {
        if let Some(aec) = &mut self.aec {
            process_buffer(aec, reference, input)
        } else {
            input.to_vec()
        }
    }
}

// Wrapper function for compatibility
#[cfg(windows)]
pub fn process_buffer_compat(
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
    fn test_webrtc_aec_creation() {
        let aec = WebRtcAec::new_stereo();
        assert!(aec.is_ok());
        if let Ok(aec) = aec {
            assert_eq!(aec.num_channels(), 2);
        }
    }

    #[cfg(windows)]
    #[test]
    fn test_digital_echo_canceller_wrapper() {
        let dec = DigitalEchoCanceller::new_stereo();
        assert!(dec.aec.is_some());
    }

    #[cfg(windows)]
    #[test]
    fn test_f32_i16_conversion() {
        let interleaved = vec![0.5f32, -0.5, 0.25, -0.25];
        let deinterleaved = f32_interleaved_to_i16_deinterleaved(&interleaved, 2);

        assert_eq!(deinterleaved.len(), 2);
        assert_eq!(deinterleaved[0].len(), 2); // L channel
        assert_eq!(deinterleaved[1].len(), 2); // R channel

        // Convert back
        let back = i16_deinterleaved_to_f32_interleaved(&deinterleaved);
        assert_eq!(back.len(), 4);

        // Check values are approximately equal (some precision loss from i16)
        for (orig, converted) in interleaved.iter().zip(back.iter()) {
            assert!((orig - converted).abs() < 0.001);
        }
    }
}
