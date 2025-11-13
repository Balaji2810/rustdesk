# WASAPI Acoustic Echo Cancellation (AEC) Implementation

## Current Status: Full AEC Implementation with Windows COM

✅ **Fully Implemented**: This implementation uses Windows COM interfaces directly via the `windows-rs` crate to provide proper WASAPI-based Acoustic Echo Cancellation (AEC) for RustDesk's client-side audio playback on Windows.

## Overview

This document describes the complete implementation of WASAPI-based Acoustic Echo Cancellation (AEC) for RustDesk's client-side audio playback on Windows, using direct COM interface access.

## What is WASAPI AEC?

WASAPI (Windows Audio Session API) provides built-in Acoustic Echo Cancellation through its audio processing features. The AEC removes echo by:
- Using the **microphone input** as a reference stream
- Processing the **speaker output** (audio playback)
- Removing any echo of the playback that appears in the microphone

## Architecture

### Before (Simple Buffer Subtraction)
```
Server Side:
- Captures speaker loopback + microphone
- Simple buffer subtraction for echo cancellation
- Mixes and sends to client

Client Side:
- Receives audio from server
- Plays audio through speakers
- No echo cancellation
```

### After (WASAPI AEC)
```
Server Side:
- Captures speaker loopback + microphone
- Keeps simple echo cancellation for backwards compatibility
- Mixes and sends to client

Client Side (Windows):
- Receives audio from server
- Uses WASAPI with AEC enabled
- References microphone input for echo cancellation
- Automatically removes echo from playback
```

## Implementation Details

### Files Modified

1. **Cargo.toml**
   - Added `wasapi = "0.14"` dependency for Windows

2. **src/client/aec_wasapi.rs** (NEW)
   - WASAPI audio handler with AEC support
   - Checks AEC availability
   - Configures microphone as reference stream
   - Manages audio rendering with echo cancellation

3. **src/client.rs**
   - Added WASAPI AEC module import
   - Updated `AudioHandler` structure with WASAPI fields
   - Modified `start_audio()` to detect and use WASAPI AEC
   - Added `start_audio_with_wasapi_aec()` for Windows
   - Updated `handle_frame()` to support WASAPI mode

4. **src/server/audio_service.rs**
   - Added comments explaining the dual echo cancellation approach
   - Kept server-side echo cancellation for backwards compatibility

### Key Components

#### 1. WasapiAecAudioHandler

Main class for WASAPI AEC audio management using Windows COM interfaces:

```rust
pub struct WasapiAecAudioHandler {
    audio_client: Option<IAudioClient>,
    render_client: Option<IAudioRenderClient>,
    sample_rate: u32,
    channels: u16,
    buffer_frame_count: u32,
    is_aec_enabled: bool,
    _com_initialized: bool,
}
```

**Key Methods:**
- `check_aec_support()` - Verifies IAudioClient3 availability for AEC
- `initialize()` - Sets up WASAPI with AEC using COM interfaces
- `try_enable_aec()` - Attempts to enable AEC with microphone reference
- `start()`/`stop()` - Controls the audio stream
- `write_data()` - Writes audio data through AEC-enabled stream

**COM Interfaces Used:**
- `IAudioClient` / `IAudioClient3` - Core WASAPI audio client
- `IAudioRenderClient` - Audio rendering
- `IMMDeviceEnumerator` - Device enumeration
- `IMMDevice` - Audio device management
- `IAcousticEchoCancellationControl` - AEC control (manually defined)

#### 2. AudioHandler Integration

The `AudioHandler` in `client.rs` now:
- Checks for WASAPI AEC support on initialization
- Falls back to cpal if AEC is unavailable
- Manages a background thread for continuous audio writing
- Maintains compatibility with non-Windows platforms

### Audio Flow

1. **Initialization**
   ```
   Client starts → check_aec_support() → initialize() → start()
   ```

2. **Playback**
   ```
   Server audio → decode → resample/rechannel → audio_buffer → WASAPI AEC → speakers
                                                                    ↑
                                                              [microphone reference]
   ```

3. **Echo Cancellation**
   ```
   Microphone captures audio → WASAPI AEC uses it as reference →
   Removes echo from speaker output → Clean audio to microphone
   ```

## Configuration

### AEC is automatically enabled when:
- Running on Windows
- WASAPI AEC is supported by the audio hardware
- Default audio devices are available

### Fallback behavior:
- If AEC is not supported → uses standard cpal audio
- If initialization fails → falls back to cpal
- Non-Windows platforms → continue using cpal

## Benefits

1. **Superior Echo Cancellation**
   - Uses Microsoft's tested AEC implementation
   - Adaptive to room acoustics
   - Hardware acceleration when available

2. **Automatic Configuration**
   - No manual setup required
   - Automatically uses default microphone as reference
   - Graceful fallback if unavailable

3. **Low CPU Usage**
   - Hardware-accelerated when supported
   - Efficient Microsoft implementation

4. **Backwards Compatibility**
   - Server-side echo cancellation still works
   - Non-Windows clients unaffected
   - Automatic detection and fallback

## Testing

### To verify WASAPI AEC is working:

1. **Check logs** for these messages:
   ```
   INFO: WASAPI AEC support detected: true
   INFO: Attempting to use WASAPI AEC for audio playback
   INFO: Successfully enabled WASAPI AEC with microphone as reference
   INFO: WASAPI AEC audio playback initialized successfully
   ```

2. **Verify echo reduction**:
   - Start a voice call
   - Play audio on the server
   - Speak into the client's microphone while audio is playing
   - Echo should be significantly reduced or eliminated

3. **Check fallback behavior**:
   - If AEC is not supported, should see:
     ```
     INFO: WASAPI AEC not supported, using cpal
     ```

### Known Limitations

1. **Windows Only**
   - WASAPI is Windows-specific
   - Other platforms continue using existing implementation

2. **Hardware Dependent**
   - Some audio hardware may not support AEC
   - Fallback to cpal ensures functionality

3. **Driver Requirements**
   - Requires up-to-date audio drivers
   - Older systems may not have AEC support

## Future Improvements

1. **Configuration Options**
   - Add user option to enable/disable AEC
   - AEC strength adjustment

2. **Linux Support**
   - Investigate PulseAudio echo cancellation modules
   - Consider WebRTC AEC library integration

3. **macOS Support**
   - Research CoreAudio echo cancellation capabilities

4. **Advanced Features**
   - Noise suppression
   - Automatic gain control
   - Voice activity detection

## Troubleshooting

### AEC Not Working

1. **Check audio drivers**
   - Update to latest audio drivers
   - Verify microphone is connected and enabled

2. **Check logs**
   - Look for "WASAPI AEC" related messages
   - Check for error messages

3. **Verify devices**
   - Ensure default microphone is set in Windows
   - Ensure default speakers are set in Windows

### Audio Quality Issues

1. **Buffer issues**
   - Check for buffer underrun/overrun messages
   - Adjust buffer sizes if needed

2. **Latency**
   - WASAPI AEC may add slight latency
   - This is normal for echo cancellation processing

## Technical Implementation Details

### COM Interface Access

The implementation uses direct Windows COM APIs via `windows-rs`:

```rust
use windows::Win32::Media::Audio::{
    IAudioClient, IAudioClient3, IAudioRenderClient,
    IMMDevice, IMMDeviceEnumerator,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize,
};
```

### AEC Enablement Process

1. **Initialize COM** - `CoInitializeEx(COINIT_MULTITHREADED)`
2. **Get Devices** - Enumerate render (speaker) and capture (microphone) devices
3. **Create Audio Client** - Activate `IAudioClient` on render device
4. **Check AEC Support** - Cast to `IAudioClient3` to verify AEC capability
5. **Enable AEC** - Use `IAcousticEchoCancellationControl` interface
6. **Start Stream** - Begin audio playback with AEC active

### IAcousticEchoCancellationControl

This interface is manually defined since it's not yet in `windows-rs`:

```rust
const IID_IACOUSTIC_ECHO_CANCELLATION_CONTROL: GUID = 
    GUID::from_u128(0xf4ade780_0a0f_11e7_93ae_92361f002671);

#[repr(C)]
pub struct IAcousticEchoCancellationControl {
    __vtable: *const IAcousticEchoCancellationControlVtbl,
}
```

The interface provides `SetEchoCancellationRenderEndpoint()` to configure the microphone as the AEC reference.

### Fallback Behavior

If AEC initialization fails:
- Logs warning message
- Falls back to standard cpal audio
- System continues to work without AEC
- Server-side echo cancellation remains active

## References

- [WASAPI Documentation](https://learn.microsoft.com/en-us/windows/win32/coreaudio/wasapi)
- [Acoustic Echo Cancellation Sample](https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/acousticechocancellation/)
- [wasapi crate documentation](https://docs.rs/wasapi/latest/wasapi/)
- [windows-rs crate](https://github.com/microsoft/windows-rs) - For direct COM interface access

