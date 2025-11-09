// both soundio and cpal use wasapi on windows and coreaudio on mac, they do not support loopback.
// libpulseaudio support loopback because pulseaudio is a standalone audio service with some
// configuration, but need to install the library and start the service on OS, not a good choice.
// windows: https://docs.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
// mac: https://github.com/mattingalls/Soundflower
// https://docs.microsoft.com/en-us/windows/win32/api/audioclient/nn-audioclient-iaudioclient
// https://github.com/ExistentialAudio/BlackHole

// if pactl not work, please run
// sudo apt-get --purge --reinstall install pulseaudio
// https://askubuntu.com/questions/403416/how-to-listen-live-sounds-from-input-from-external-sound-card
// https://wiki.debian.org/audio-loopback
// https://github.com/krruzic/pulsectl

use super::*;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
use hbb_common::anyhow::anyhow;
use magnum_opus::{Application::*, Channels::*, Encoder};
use std::sync::atomic::{AtomicBool, Ordering};

pub const NAME: &'static str = "audio";
pub const AUDIO_DATA_SIZE_U8: usize = 960 * 4; // 10ms in 48000 stereo
static RESTARTING: AtomicBool = AtomicBool::new(false);

lazy_static::lazy_static! {
    static ref VOICE_CALL_INPUT_DEVICE: Arc::<Mutex::<Option<String>>> = Default::default();
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn new() -> GenericService {
    let svc = EmptyExtraFieldService::new(NAME.to_owned(), true);
    GenericService::repeat::<cpal_impl::State, _, _>(&svc.clone(), 33, cpal_impl::run);
    svc.sp
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn new() -> GenericService {
    let svc = EmptyExtraFieldService::new(NAME.to_owned(), true);
    GenericService::run(&svc.clone(), pa_impl::run);
    svc.sp
}

#[inline]
pub fn get_voice_call_input_device() -> Option<String> {
    VOICE_CALL_INPUT_DEVICE.lock().unwrap().clone()
}

#[inline]
pub fn set_voice_call_input_device(device: Option<String>, set_if_present: bool) {
    if !set_if_present && VOICE_CALL_INPUT_DEVICE.lock().unwrap().is_some() {
        return;
    }

    if *VOICE_CALL_INPUT_DEVICE.lock().unwrap() == device {
        return;
    }
    *VOICE_CALL_INPUT_DEVICE.lock().unwrap() = device;
    restart();
}

#[inline]
fn get_audio_input() -> String {
    VOICE_CALL_INPUT_DEVICE
        .lock()
        .unwrap()
        .clone()
        .unwrap_or(Config::get_option("audio-input"))
}

pub fn restart() {
    log::info!("restart the audio service, freezing now...");
    if RESTARTING.load(Ordering::SeqCst) {
        return;
    }
    RESTARTING.store(true, Ordering::SeqCst);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod pa_impl {
    use super::*;

    // SAFETY: constrains of hbb_common::mem::aligned_u8_vec must be held
    unsafe fn align_to_32(data: Vec<u8>) -> Vec<u8> {
        if (data.as_ptr() as usize & 3) == 0 {
            return data;
        }

        let mut buf = vec![];
        buf = unsafe { hbb_common::mem::aligned_u8_vec(data.len(), 4) };
        buf.extend_from_slice(data.as_ref());
        buf
    }

    #[tokio::main(flavor = "current_thread")]
    pub async fn run(sp: EmptyExtraFieldService) -> ResultType<()> {
        hbb_common::sleep(0.1).await; // one moment to wait for _pa ipc
        RESTARTING.store(false, Ordering::SeqCst);
        #[cfg(target_os = "linux")]
        let mut stream = crate::ipc::connect(1000, "_pa").await?;
        unsafe {
            AUDIO_ZERO_COUNT = 0;
        }
        let mut encoder = Encoder::new(crate::platform::PA_SAMPLE_RATE, Stereo, LowDelay)?;
        #[cfg(target_os = "linux")]
        allow_err!(
            stream
                .send(&crate::ipc::Data::Config((
                    "audio-input".to_owned(),
                    Some(super::get_audio_input())
                )))
                .await
        );
        #[cfg(target_os = "linux")]
        let zero_audio_frame: Vec<f32> = vec![0.; AUDIO_DATA_SIZE_U8 / 4];
        #[cfg(target_os = "android")]
        let mut android_data = vec![];
        while sp.ok() && !RESTARTING.load(Ordering::SeqCst) {
            sp.snapshot(|sps| {
                sps.send(create_format_msg(crate::platform::PA_SAMPLE_RATE, 2));
                Ok(())
            })?;

            #[cfg(target_os = "linux")]
            if let Ok(data) = stream.next_raw().await {
                if data.len() == 0 {
                    send_f32(&zero_audio_frame, &mut encoder, &sp);
                    continue;
                }

                if data.len() != AUDIO_DATA_SIZE_U8 {
                    continue;
                }

                let data = unsafe { align_to_32(data.into()) };
                let data = unsafe {
                    std::slice::from_raw_parts::<f32>(data.as_ptr() as _, data.len() / 4)
                };
                send_f32(data, &mut encoder, &sp);
            }

            #[cfg(target_os = "android")]
            if scrap::android::ffi::get_audio_raw(&mut android_data, &mut vec![]).is_some() {
                let data = unsafe {
                    android_data = align_to_32(android_data);
                    std::slice::from_raw_parts::<f32>(
                        android_data.as_ptr() as _,
                        android_data.len() / 4,
                    )
                };
                send_f32(data, &mut encoder, &sp);
            } else {
                hbb_common::sleep(0.1).await;
            }
        }
        Ok(())
    }
}

#[inline]
#[cfg(feature = "screencapturekit")]
pub fn is_screen_capture_kit_available() -> bool {
    cpal::available_hosts()
        .iter()
        .any(|host| *host == cpal::HostId::ScreenCaptureKit)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod cpal_impl {
    use self::service::{Reset, ServiceSwap};
    use super::*;
    use cpal::{
        traits::{DeviceTrait, HostTrait, StreamTrait},
        BufferSize, Device, Host, InputCallbackInfo, StreamConfig, SupportedStreamConfig,
    };

    lazy_static::lazy_static! {
        static ref HOST: Host = cpal::default_host();
        static ref INPUT_BUFFER: Arc<Mutex<std::collections::VecDeque<f32>>> = Default::default();
        #[cfg(windows)]
        static ref LOOPBACK_BUFFER: Arc<Mutex<std::collections::VecDeque<f32>>> = Default::default();
        #[cfg(windows)]
        static ref MIC_BUFFER: Arc<Mutex<std::collections::VecDeque<f32>>> = Default::default();
    }

    #[cfg(feature = "screencapturekit")]
    lazy_static::lazy_static! {
        static ref HOST_SCREEN_CAPTURE_KIT: Result<Host, cpal::HostUnavailable> = cpal::host_from_id(cpal::HostId::ScreenCaptureKit);
    }

    #[derive(Default)]
    pub struct State {
        stream: Option<(Box<dyn StreamTrait>, Arc<Message>)>,
        #[cfg(windows)]
        loopback_stream: Option<Box<dyn StreamTrait>>,
        #[cfg(windows)]
        mic_stream: Option<Box<dyn StreamTrait>>,
    }

    impl super::service::Reset for State {
        fn reset(&mut self) {
            self.stream.take();
            #[cfg(windows)]
            {
                self.loopback_stream.take();
                self.mic_stream.take();
            }
        }
    }

    fn run_restart(sp: EmptyExtraFieldService, state: &mut State) -> ResultType<()> {
        state.reset();
        sp.snapshot(|_sps: ServiceSwap<_>| Ok(()))?;
        #[cfg(windows)]
        {
            // On Windows, play_mixed returns both streams
            let (loopback_stream, mic_stream, format) = play_mixed(&sp)?;
            state.mic_stream = Some(mic_stream);
            // Store the loopback stream in the main stream field for compatibility
            // The mic stream is stored separately and both are kept alive
            state.stream = Some((loopback_stream, format));
        }
        #[cfg(not(windows))]
        {
            match &state.stream {
                None => {
                    state.stream = Some(play(&sp)?);
                }
                _ => {}
            }
        }
        if let Some((_, format)) = &state.stream {
            sp.send_shared(format.clone());
        }
        RESTARTING.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn run_serv_snapshot(sp: EmptyExtraFieldService, state: &mut State) -> ResultType<()> {
        sp.snapshot(|sps| {
            #[cfg(windows)]
            {
                if state.stream.is_none() || state.mic_stream.is_none() {
                    let (loopback_stream, mic_stream, format) = play_mixed(&sp)?;
                    state.mic_stream = Some(mic_stream);
                    state.stream = Some((loopback_stream, format));
                }
            }
            #[cfg(not(windows))]
            {
                match &state.stream {
                    None => {
                        state.stream = Some(play(&sp)?);
                    }
                    _ => {}
                }
            }
            if let Some((_, format)) = &state.stream {
                sps.send_shared(format.clone());
            }
            Ok(())
        })?;
        Ok(())
    }

    pub fn run(sp: EmptyExtraFieldService, state: &mut State) -> ResultType<()> {
        if !RESTARTING.load(Ordering::SeqCst) {
            run_serv_snapshot(sp, state)
        } else {
            run_restart(sp, state)
        }
    }

    fn send(
        data: Vec<f32>,
        sample_rate0: u32,
        sample_rate: u32,
        device_channel: u16,
        encode_channel: u16,
        encoder: &mut Encoder,
        sp: &GenericService,
    ) {
        let mut data = data;
        if sample_rate0 != sample_rate {
            data = crate::common::audio_resample(&data, sample_rate0, sample_rate, device_channel);
        }
        if device_channel != encode_channel {
            data = crate::common::audio_rechannel(
                data,
                sample_rate,
                sample_rate,
                device_channel,
                encode_channel,
            )
        }
        send_f32(&data, encoder, sp);
    }

    #[cfg(windows)]
    fn mix_and_send(
        loopback_data: Vec<f32>,
        loopback_sample_rate0: u32,
        loopback_device_channel: u16,
        mic_data: Vec<f32>,
        mic_sample_rate0: u32,
        mic_device_channel: u16,
        sample_rate: u32,
        encode_channel: u16,
        encoder: Arc<Mutex<Encoder>>,
        sp: &GenericService,
    ) {
        // Resample both sources to target sample rate
        let mut loopback_resampled = if loopback_sample_rate0 != sample_rate {
            crate::common::audio_resample(&loopback_data, loopback_sample_rate0, sample_rate, loopback_device_channel)
        } else {
            loopback_data
        };
        
        let mut mic_resampled = if mic_sample_rate0 != sample_rate {
            crate::common::audio_resample(&mic_data, mic_sample_rate0, sample_rate, mic_device_channel)
        } else {
            mic_data
        };

        // Convert both to target channel count (stereo)
        loopback_resampled = if loopback_device_channel != encode_channel {
            crate::common::audio_rechannel(
                loopback_resampled,
                sample_rate,
                sample_rate,
                loopback_device_channel,
                encode_channel,
            )
        } else {
            loopback_resampled
        };

        mic_resampled = if mic_device_channel != encode_channel {
            crate::common::audio_rechannel(
                mic_resampled,
                sample_rate,
                sample_rate,
                mic_device_channel,
                encode_channel,
            )
        } else {
            mic_resampled
        };

        // Ensure both have the same length (take minimum)
        let min_len = std::cmp::min(loopback_resampled.len(), mic_resampled.len());
        if min_len == 0 {
            return;
        }

        // Mix the two sources together (simple addition with normalization to prevent clipping)
        let mut mixed = Vec::with_capacity(min_len);
        for i in 0..min_len {
            let loopback_val = loopback_resampled.get(i).copied().unwrap_or(0.0);
            let mic_val = mic_resampled.get(i).copied().unwrap_or(0.0);
            // Mix with 0.5 gain each to prevent clipping, then normalize if needed
            let mixed_val = (loopback_val * 0.5) + (mic_val * 0.5);
            // Clamp to [-1.0, 1.0] range
            mixed.push(mixed_val.max(-1.0).min(1.0));
        }

        // Lock encoder and send
        let mut encoder_guard = encoder.lock().unwrap();
        send_f32(&mixed, &mut *encoder_guard, sp);
    }

    #[cfg(feature = "screencapturekit")]
    fn get_device() -> ResultType<(Device, SupportedStreamConfig)> {
        let audio_input = super::get_audio_input();
        if !audio_input.is_empty() {
            return get_audio_input(&audio_input);
        }
        if !is_screen_capture_kit_available() {
            return get_audio_input("");
        }
        let device = HOST_SCREEN_CAPTURE_KIT
            .as_ref()?
            .default_input_device()
            .with_context(|| "Failed to get default input device for loopback")?;
        let format = device
            .default_input_config()
            .map_err(|e| anyhow!(e))
            .with_context(|| "Failed to get input output format")?;
        log::info!("Default input format: {:?}", format);
        Ok((device, format))
    }

    #[cfg(windows)]
    fn get_device() -> ResultType<(Device, SupportedStreamConfig)> {
        let audio_input = super::get_audio_input();
        if !audio_input.is_empty() {
            return get_audio_input(&audio_input);
        }
        let device = HOST
            .default_output_device()
            .with_context(|| "Failed to get default output device for loopback")?;
        log::info!(
            "Default output device: {}",
            device.name().unwrap_or("".to_owned())
        );
        let format = device
            .default_output_config()
            .map_err(|e| anyhow!(e))
            .with_context(|| "Failed to get default output format")?;
        log::info!("Default output format: {:?}", format);
        Ok((device, format))
    }

    #[cfg(windows)]
    fn get_loopback_device() -> ResultType<(Device, SupportedStreamConfig)> {
        let device = HOST
            .default_output_device()
            .with_context(|| "Failed to get default output device for loopback")?;
        log::info!(
            "Loopback device: {}",
            device.name().unwrap_or("".to_owned())
        );
        let format = device
            .default_output_config()
            .map_err(|e| anyhow!(e))
            .with_context(|| "Failed to get default output format")?;
        log::info!("Loopback format: {:?}", format);
        Ok((device, format))
    }

    #[cfg(windows)]
    fn get_mic_device() -> ResultType<(Device, SupportedStreamConfig)> {
        let audio_input = super::get_audio_input();
        let device = if !audio_input.is_empty() {
            // Try to find the specified device
            let mut found_device = None;
            for d in HOST
                .devices()
                .with_context(|| "Failed to get audio devices")?
            {
                if d.name().unwrap_or("".to_owned()) == audio_input {
                    found_device = Some(d);
                    break;
                }
            }
            found_device.unwrap_or_else(|| {
                HOST.default_input_device()
                    .with_context(|| "Failed to get default input device")
                    .unwrap()
            })
        } else {
            HOST.default_input_device()
                .with_context(|| "Failed to get default input device")?
        };
        log::info!(
            "Mic device: {}",
            device.name().unwrap_or("".to_owned())
        );
        let format = device
            .default_input_config()
            .map_err(|e| anyhow!(e))
            .with_context(|| "Failed to get default input format")?;
        log::info!("Mic format: {:?}", format);
        Ok((device, format))
    }

    #[cfg(not(any(windows, feature = "screencapturekit")))]
    fn get_device() -> ResultType<(Device, SupportedStreamConfig)> {
        let audio_input = super::get_audio_input();
        get_audio_input(&audio_input)
    }

    fn get_audio_input(audio_input: &str) -> ResultType<(Device, SupportedStreamConfig)> {
        let mut device = None;
        #[cfg(feature = "screencapturekit")]
        if !audio_input.is_empty() && is_screen_capture_kit_available() {
            for d in HOST_SCREEN_CAPTURE_KIT
                .as_ref()?
                .devices()
                .with_context(|| "Failed to get audio devices")?
            {
                if d.name().unwrap_or("".to_owned()) == audio_input {
                    device = Some(d);
                    break;
                }
            }
        }
        if device.is_none() && !audio_input.is_empty() {
            for d in HOST
                .devices()
                .with_context(|| "Failed to get audio devices")?
            {
                if d.name().unwrap_or("".to_owned()) == audio_input {
                    device = Some(d);
                    break;
                }
            }
        }
        let device = device.unwrap_or(
            HOST.default_input_device()
                .with_context(|| "Failed to get default input device for loopback")?,
        );
        log::info!("Input device: {}", device.name().unwrap_or("".to_owned()));
        let format = device
            .default_input_config()
            .map_err(|e| anyhow!(e))
            .with_context(|| "Failed to get default input format")?;
        log::info!("Default input format: {:?}", format);
        Ok((device, format))
    }

    #[cfg(windows)]
    fn play_mixed(sp: &GenericService) -> ResultType<(Box<dyn StreamTrait>, Box<dyn StreamTrait>, Arc<Message>)> {
        use cpal::SampleFormat::*;
        let (loopback_device, loopback_config) = get_loopback_device()?;
        let (mic_device, mic_config) = get_mic_device()?;
        let sp = sp.clone();
        
        // Use 48000 Hz as the target sample rate for mixing
        let sample_rate = 48000;
        let ch = Stereo; // Always use stereo for mixed output
        
        // Get sample rates and channels for both sources
        let loopback_sample_rate_0 = loopback_config.sample_rate().0;
        let loopback_device_channel = loopback_config.channels();
        let mic_sample_rate_0 = mic_config.sample_rate().0;
        let mic_device_channel = mic_config.channels();
        
        // Create encoder for mixed output (wrapped in Arc<Mutex> for sharing)
        let encoder = Arc::new(Mutex::new(Encoder::new(sample_rate, ch, LowDelay)?));
        let encode_channel = ch as u16;
        
        // Calculate frame sizes
        let frame_size = sample_rate as usize / 100; // 10 ms
        let encode_len = frame_size * encode_channel as usize;
        let loopback_rechannel_len = encode_len * loopback_device_channel as usize / encode_channel as usize;
        let mic_rechannel_len = encode_len * mic_device_channel as usize / encode_channel as usize;
        
        // Clear buffers
        LOOPBACK_BUFFER.lock().unwrap().clear();
        MIC_BUFFER.lock().unwrap().clear();
        
        unsafe {
            AUDIO_ZERO_COUNT = 0;
        }
        
        // Create loopback stream
        let loopback_stream = match loopback_config.sample_format() {
            I8 => build_mixing_stream::<i8>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I16 => build_mixing_stream::<i16>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I32 => build_mixing_stream::<i32>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I64 => build_mixing_stream::<i64>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U8 => build_mixing_stream::<u8>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U16 => build_mixing_stream::<u16>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U32 => build_mixing_stream::<u32>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U64 => build_mixing_stream::<u64>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            F32 => build_mixing_stream::<f32>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            F64 => build_mixing_stream::<f64>(loopback_device, &loopback_config, sp.clone(), true, loopback_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            f => bail!("unsupported loopback audio format: {:?}", f),
        };
        
        // Create mic stream
        let mic_stream = match mic_config.sample_format() {
            I8 => build_mixing_stream::<i8>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I16 => build_mixing_stream::<i16>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I32 => build_mixing_stream::<i32>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            I64 => build_mixing_stream::<i64>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U8 => build_mixing_stream::<u8>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U16 => build_mixing_stream::<u16>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U32 => build_mixing_stream::<u32>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            U64 => build_mixing_stream::<u64>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            F32 => build_mixing_stream::<f32>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            F64 => build_mixing_stream::<f64>(mic_device, &mic_config, sp.clone(), false, mic_rechannel_len, loopback_sample_rate_0, loopback_device_channel, mic_sample_rate_0, mic_device_channel, sample_rate, encode_channel, encoder.clone())?,
            f => bail!("unsupported mic audio format: {:?}", f),
        };
        
        loopback_stream.play()?;
        mic_stream.play()?;
        
        // Return both streams
        Ok((
            Box::new(loopback_stream),
            Box::new(mic_stream),
            Arc::new(create_format_msg(sample_rate, ch as _)),
        ))
    }

    fn play(sp: &GenericService) -> ResultType<(Box<dyn StreamTrait>, Arc<Message>)> {
        #[cfg(windows)]
        {
            // On Windows, use mixing mode to combine loopback and mic
            return play_mixed(sp);
        }
        
        #[cfg(not(windows))]
        {
            use cpal::SampleFormat::*;
            let (device, config) = get_device()?;
            let sp = sp.clone();
            // Sample rate must be one of 8000, 12000, 16000, 24000, or 48000.
            let sample_rate_0 = config.sample_rate().0;
            let sample_rate = if sample_rate_0 < 12000 {
                8000
            } else if sample_rate_0 < 16000 {
                12000
            } else if sample_rate_0 < 24000 {
                16000
            } else if sample_rate_0 < 48000 {
                24000
            } else {
                48000
            };
            let ch = if config.channels() > 1 { Stereo } else { Mono };
            let stream = match config.sample_format() {
                I8 => build_input_stream::<i8>(device, &config, sp, sample_rate, ch)?,
                I16 => build_input_stream::<i16>(device, &config, sp, sample_rate, ch)?,
                I32 => build_input_stream::<i32>(device, &config, sp, sample_rate, ch)?,
                I64 => build_input_stream::<i64>(device, &config, sp, sample_rate, ch)?,
                U8 => build_input_stream::<u8>(device, &config, sp, sample_rate, ch)?,
                U16 => build_input_stream::<u16>(device, &config, sp, sample_rate, ch)?,
                U32 => build_input_stream::<u32>(device, &config, sp, sample_rate, ch)?,
                U64 => build_input_stream::<u64>(device, &config, sp, sample_rate, ch)?,
                F32 => build_input_stream::<f32>(device, &config, sp, sample_rate, ch)?,
                F64 => build_input_stream::<f64>(device, &config, sp, sample_rate, ch)?,
                f => bail!("unsupported audio format: {:?}", f),
            };
            stream.play()?;
            Ok((
                Box::new(stream),
                Arc::new(create_format_msg(sample_rate, ch as _)),
            ))
        }
    }

    fn build_input_stream<T>(
        device: cpal::Device,
        config: &cpal::SupportedStreamConfig,
        sp: GenericService,
        sample_rate: u32,
        encode_channel: magnum_opus::Channels,
    ) -> ResultType<cpal::Stream>
    where
        T: cpal::SizedSample + dasp::sample::ToSample<f32>,
    {
        let err_fn = move |err| {
            // too many UnknownErrno, will improve later
            log::trace!("an error occurred on stream: {}", err);
        };
        let sample_rate_0 = config.sample_rate().0;
        log::debug!("Audio sample rate : {}", sample_rate);
        unsafe {
            AUDIO_ZERO_COUNT = 0;
        }
        let device_channel = config.channels();
        let mut encoder = Encoder::new(sample_rate, encode_channel, LowDelay)?;
        // https://www.opus-codec.org/docs/html_api/group__opusencoder.html#gace941e4ef26ed844879fde342ffbe546
        // https://chromium.googlesource.com/chromium/deps/opus/+/1.1.1/include/opus.h
        // Do not set `frame_size = sample_rate as usize / 100;`
        // Because we find `sample_rate as usize / 100` will cause encoder error in `encoder.encode_vec_float()` sometimes.
        // https://github.com/xiph/opus/blob/2554a89e02c7fc30a980b4f7e635ceae1ecba5d6/src/opus_encoder.c#L725
        let frame_size = sample_rate_0 as usize / 100; // 10 ms
        let encode_len = frame_size * encode_channel as usize;
        let rechannel_len = encode_len * device_channel as usize / encode_channel as usize;
        INPUT_BUFFER.lock().unwrap().clear();
        let timeout = None;
        let stream_config = StreamConfig {
            channels: device_channel,
            sample_rate: config.sample_rate(),
            buffer_size: BufferSize::Default,
        };
        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[T], _: &InputCallbackInfo| {
                let buffer: Vec<f32> = data.iter().map(|s| T::to_sample(*s)).collect();
                let mut lock = INPUT_BUFFER.lock().unwrap();
                lock.extend(buffer);
                while lock.len() >= rechannel_len {
                    let frame: Vec<f32> = lock.drain(0..rechannel_len).collect();
                    send(
                        frame,
                        sample_rate_0,
                        sample_rate,
                        device_channel,
                        encode_channel as _,
                        &mut encoder,
                        &sp,
                    );
                }
            },
            err_fn,
            timeout,
        )?;
        Ok(stream)
    }

    #[cfg(windows)]
    fn build_mixing_stream<T>(
        device: cpal::Device,
        config: &cpal::SupportedStreamConfig,
        sp: GenericService,
        is_loopback: bool,
        rechannel_len: usize,
        loopback_sample_rate_0: u32,
        loopback_device_channel: u16,
        mic_sample_rate_0: u32,
        mic_device_channel: u16,
        target_sample_rate: u32,
        encode_channel: u16,
        encoder: Arc<Mutex<Encoder>>,
    ) -> ResultType<cpal::Stream>
    where
        T: cpal::SizedSample + dasp::sample::ToSample<f32>,
    {
        let err_fn = move |err| {
            log::trace!("an error occurred on stream: {}", err);
        };
        let timeout = None;
        let device_channel = config.channels();
        let sample_rate_0 = config.sample_rate().0;
        let stream_config = StreamConfig {
            channels: device_channel,
            sample_rate: config.sample_rate(),
            buffer_size: BufferSize::Default,
        };
        
        let sp_clone = sp.clone();
        let encoder_clone = encoder.clone();
        
        // Calculate frame sizes for mixing
        let frame_size = target_sample_rate as usize / 100; // 10 ms
        let encode_len = frame_size * encode_channel as usize;
        // Calculate rechannel_len for this specific device
        let this_device_rechannel_len = encode_len * device_channel as usize / encode_channel as usize;
        // Calculate rechannel_len for the other device
        let other_device_rechannel_len = if is_loopback {
            encode_len * mic_device_channel as usize / encode_channel as usize
        } else {
            encode_len * loopback_device_channel as usize / encode_channel as usize
        };
        
        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[T], _: &InputCallbackInfo| {
                let buffer: Vec<f32> = data.iter().map(|s| T::to_sample(*s)).collect();
                let target_buffer = if is_loopback {
                    &LOOPBACK_BUFFER
                } else {
                    &MIC_BUFFER
                };
                let mut lock = target_buffer.lock().unwrap();
                lock.extend(buffer);
                drop(lock);
                
                // Trigger mixing when we have enough data in both buffers
                let loopback_lock = LOOPBACK_BUFFER.lock().unwrap();
                let mic_lock = MIC_BUFFER.lock().unwrap();
                
                // Calculate the required lengths for both buffers
                let loopback_rechannel_len = if is_loopback {
                    this_device_rechannel_len
                } else {
                    other_device_rechannel_len
                };
                let mic_rechannel_len = if is_loopback {
                    other_device_rechannel_len
                } else {
                    this_device_rechannel_len
                };
                
                // Check if we have enough data in both buffers
                if loopback_lock.len() >= loopback_rechannel_len && mic_lock.len() >= mic_rechannel_len {
                    drop(loopback_lock);
                    drop(mic_lock);
                    
                    // Extract frames from both buffers
                    let mut loopback_lock = LOOPBACK_BUFFER.lock().unwrap();
                    let mut mic_lock = MIC_BUFFER.lock().unwrap();
                    
                    let loopback_frame: Vec<f32> = loopback_lock.drain(0..loopback_rechannel_len).collect();
                    let mic_frame: Vec<f32> = mic_lock.drain(0..mic_rechannel_len).collect();
                    
                    drop(loopback_lock);
                    drop(mic_lock);
                    
                    // Mix and send
                    mix_and_send(
                        loopback_frame,
                        loopback_sample_rate_0,
                        loopback_device_channel,
                        mic_frame,
                        mic_sample_rate_0,
                        mic_device_channel,
                        target_sample_rate,
                        encode_channel,
                        encoder_clone.clone(),
                        &sp_clone,
                    );
                }
            },
            err_fn,
            timeout,
        )?;
        Ok(stream)
    }
}

fn create_format_msg(sample_rate: u32, channels: u16) -> Message {
    let format = AudioFormat {
        sample_rate,
        channels: channels as _,
        ..Default::default()
    };
    let mut misc = Misc::new();
    misc.set_audio_format(format);
    let mut msg = Message::new();
    msg.set_misc(misc);
    msg
}

// use AUDIO_ZERO_COUNT for the Noise(Zero) Gate Attack Time
// every audio data length is set to 480
// MAX_AUDIO_ZERO_COUNT=800 is similar as Gate Attack Time 3~5s(Linux) || 6~8s(Windows)
const MAX_AUDIO_ZERO_COUNT: u16 = 800;
static mut AUDIO_ZERO_COUNT: u16 = 0;

fn send_f32(data: &[f32], encoder: &mut Encoder, sp: &GenericService) {
    if data.iter().filter(|x| **x != 0.).next().is_some() {
        unsafe {
            AUDIO_ZERO_COUNT = 0;
        }
    } else {
        unsafe {
            if AUDIO_ZERO_COUNT > MAX_AUDIO_ZERO_COUNT {
                if AUDIO_ZERO_COUNT == MAX_AUDIO_ZERO_COUNT + 1 {
                    log::debug!("Audio Zero Gate Attack");
                    AUDIO_ZERO_COUNT += 1;
                }
                return;
            }
            AUDIO_ZERO_COUNT += 1;
        }
    }
    #[cfg(target_os = "android")]
    {
        // the permitted opus data size are 120, 240, 480, 960, 1920, and 2880
        // if data size is bigger than BATCH_SIZE, AND is an integer multiple of BATCH_SIZE
        // then upload in batches
        const BATCH_SIZE: usize = 960;
        let input_size = data.len();
        if input_size > BATCH_SIZE && input_size % BATCH_SIZE == 0 {
            let n = input_size / BATCH_SIZE;
            for i in 0..n {
                match encoder
                    .encode_vec_float(&data[i * BATCH_SIZE..(i + 1) * BATCH_SIZE], BATCH_SIZE)
                {
                    Ok(data) => {
                        let mut msg_out = Message::new();
                        msg_out.set_audio_frame(AudioFrame {
                            data: data.into(),
                            ..Default::default()
                        });
                        sp.send(msg_out);
                    }
                    Err(_) => {}
                }
            }
        } else {
            log::debug!("invalid audio data size:{} ", input_size);
            return;
        }
    }

    #[cfg(not(target_os = "android"))]
    match encoder.encode_vec_float(data, data.len() * 6) {
        Ok(data) => {
            let mut msg_out = Message::new();
            msg_out.set_audio_frame(AudioFrame {
                data: data.into(),
                ..Default::default()
            });
            sp.send(msg_out);
        }
        Err(_) => {}
    }
}
