// WASAPI-based Acoustic Echo Cancellation (AEC) module for Windows
// This module provides AEC functionality using Windows' native WASAPI audio API

use hbb_common::{anyhow::Context, bail, log, ResultType};
use std::sync::{Arc, Mutex};
use wasapi::*;

pub struct WasapiAecAudioHandler {
    audio_client: Option<AudioClient>,
    render_client: Option<AudioRenderClient>,
    capture_client: Option<AudioCaptureClient>,
    format: Option<WaveFormat>,
    buffer_frame_count: u32,
    is_aec_supported: bool,
}

impl WasapiAecAudioHandler {
    pub fn new() -> Self {
        Self {
            audio_client: None,
            render_client: None,
            capture_client: None,
            format: None,
            buffer_frame_count: 0,
            is_aec_supported: false,
        }
    }

    /// Check if AEC is supported on this system
    pub fn check_aec_support() -> ResultType<bool> {
        let device_enumerator = DeviceEnumerator::new()?;
        let device = device_enumerator.get_default_render_device()?;
        let mut audio_client = device.get_iaudioclient()?;
        
        match audio_client.is_aec_supported() {
            Ok(supported) => {
                log::info!("WASAPI AEC support detected: {}", supported);
                Ok(supported)
            }
            Err(e) => {
                log::warn!("Could not check AEC support: {}", e);
                Ok(false)
            }
        }
    }

    /// Initialize WASAPI with AEC enabled
    /// Uses the default microphone as reference for echo cancellation
    pub fn initialize(&mut self, sample_rate: u32, channels: u16) -> ResultType<()> {
        log::info!("Initializing WASAPI AEC with sample_rate={}, channels={}", sample_rate, channels);
        
        // Get the default render device (speakers/headphones)
        let device_enumerator = DeviceEnumerator::new()
            .with_context(|| "Failed to create device enumerator")?;
        let render_device = device_enumerator.get_default_render_device()
            .with_context(|| "Failed to get default render device")?;
        
        // Get audio client for render device
        let mut audio_client = render_device.get_iaudioclient()
            .with_context(|| "Failed to get audio client")?;
        
        // Check if AEC is supported
        self.is_aec_supported = audio_client.is_aec_supported().unwrap_or(false);
        
        if !self.is_aec_supported {
            log::warn!("WASAPI AEC is not supported on this system, falling back to non-AEC mode");
            bail!("AEC not supported");
        }
        
        // Create wave format for the audio stream
        let wave_format = WaveFormat::new(
            32, // bits per sample (f32)
            32, // bits per sample
            &SampleType::Float,
            sample_rate as usize,
            channels as usize,
            None,
        );
        
        // Initialize audio client in shared mode with event-driven buffer
        let blockalign = wave_format.get_blockalign();
        let (_, period) = audio_client.get_periods()?;
        
        audio_client.initialize_client(
            &wave_format,
            period as i64,
            &Direction::Render,
            &ShareMode::Shared,
            true, // use event
        )?;
        
        // Get buffer frame count
        self.buffer_frame_count = audio_client.get_bufferframecount()?;
        
        // Get render client for writing audio data
        let render_client = audio_client.get_audiorenderclient()?;
        
        // Enable AEC by setting the capture device as reference
        if self.is_aec_supported {
            if let Ok(aec_control) = audio_client.get_aec_control() {
                // Get the default capture device (microphone)
                if let Ok(capture_device) = device_enumerator.get_default_capture_device() {
                    match aec_control.set_echo_cancellation_render_endpoint(&capture_device) {
                        Ok(_) => {
                            log::info!("Successfully enabled WASAPI AEC with microphone as reference");
                        }
                        Err(e) => {
                            log::warn!("Failed to set AEC reference endpoint: {}", e);
                        }
                    }
                } else {
                    log::warn!("Could not get default capture device for AEC reference");
                }
            } else {
                log::warn!("Could not get AEC control interface");
            }
        }
        
        self.audio_client = Some(audio_client);
        self.render_client = Some(render_client);
        self.format = Some(wave_format);
        
        log::info!("WASAPI AEC initialization complete, buffer_frame_count={}", self.buffer_frame_count);
        Ok(())
    }

    /// Start the audio stream
    pub fn start(&mut self) -> ResultType<()> {
        if let Some(ref mut audio_client) = self.audio_client {
            audio_client.start_stream()?;
            log::info!("WASAPI AEC audio stream started");
            Ok(())
        } else {
            bail!("Audio client not initialized")
        }
    }

    /// Stop the audio stream
    pub fn stop(&mut self) -> ResultType<()> {
        if let Some(ref mut audio_client) = self.audio_client {
            audio_client.stop_stream()?;
            log::info!("WASAPI AEC audio stream stopped");
            Ok(())
        } else {
            bail!("Audio client not initialized")
        }
    }

    /// Write audio data to the render stream
    /// Audio will be processed through AEC if enabled
    pub fn write_data(&mut self, data: &[f32]) -> ResultType<usize> {
        if self.render_client.is_none() || self.audio_client.is_none() {
            bail!("Audio client not initialized");
        }
        
        let render_client = self.render_client.as_mut().unwrap();
        let audio_client = self.audio_client.as_mut().unwrap();
        
        // Get available buffer space
        let padding = audio_client.get_current_padding()?;
        let available = self.buffer_frame_count.saturating_sub(padding);
        
        if available == 0 {
            return Ok(0);
        }
        
        // Calculate how many frames we can write
        let channels = self.format.as_ref().unwrap().get_nchannels();
        let frames_to_write = (data.len() / channels).min(available as usize);
        
        if frames_to_write == 0 {
            return Ok(0);
        }
        
        // Get buffer from render client
        let buffer_size = frames_to_write * channels;
        let mut buffer = render_client.get_buffer(frames_to_write as u32)?;
        
        // Copy data to buffer
        let copy_size = buffer_size.min(data.len());
        unsafe {
            let buffer_slice = std::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut f32,
                buffer_size,
            );
            buffer_slice[..copy_size].copy_from_slice(&data[..copy_size]);
            
            // Fill remaining with silence if needed
            if copy_size < buffer_size {
                buffer_slice[copy_size..].fill(0.0);
            }
        }
        
        // Release buffer
        render_client.release_buffer(frames_to_write as u32, None)?;
        
        Ok(copy_size)
    }

    /// Get the configured sample rate
    pub fn get_sample_rate(&self) -> Option<u32> {
        self.format.as_ref().map(|f| f.get_samplespersec() as u32)
    }

    /// Get the configured channel count
    pub fn get_channels(&self) -> Option<u16> {
        self.format.as_ref().map(|f| f.get_nchannels() as u16)
    }

    /// Check if AEC is currently active
    pub fn is_aec_active(&self) -> bool {
        self.is_aec_supported && self.audio_client.is_some()
    }

    /// Get buffer frame count
    pub fn get_buffer_frame_count(&self) -> u32 {
        self.buffer_frame_count
    }
}

impl Drop for WasapiAecAudioHandler {
    fn drop(&mut self) {
        let _ = self.stop();
        log::debug!("WASAPI AEC audio handler dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aec_support_check() {
        // Just check that the function doesn't panic
        let _ = WasapiAecAudioHandler::check_aec_support();
    }
}

