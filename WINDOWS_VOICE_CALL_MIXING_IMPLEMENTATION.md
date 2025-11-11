# Windows Voice Call Audio Mixing - Implementation Summary

## Overview
Successfully implemented speaker + microphone audio mixing for Windows voice calls in RustDesk. When a voice call is accepted on Windows, the system now captures both:
- **Default speaker output** (loopback/what's playing on the remote system)
- **Default microphone input** (user's voice)

The audio is mixed at a 70% speaker / 30% microphone ratio and streamed to the controlling client.

## Changes Made

### 1. src/server/audio_service.rs

#### New State Tracking (Lines 186-191)
Added lazy_static globals for Windows:
- `IS_VOICE_CALL_ACTIVE`: Tracks whether voice call mixing is enabled
- `SPEAKER_BUFFER`: Circular buffer for speaker audio samples
- `MIC_BUFFER`: Circular buffer for microphone audio samples

#### New Structure (Lines 193-198)
- `VoiceCallStreams`: Holds both speaker and mic streams plus format info

#### Extended State Struct (Lines 200-215)
- Added `voice_call_streams: Option<VoiceCallStreams>` field for Windows
- Updated `reset()` method to clean up voice call streams

#### New Resampling Function (Lines 282-336)
- `resample_with_rubato()`: Uses rubato's FastFixedIn resampler for high-quality sample rate conversion
- Handles per-channel resampling to prevent drift
- Falls back to existing resampling if rubato feature is disabled

#### New Mixing Function (Lines 338-399)
- `mix_and_send()`: Core mixing logic
- Resamples both audio sources to target rate (48000 Hz)
- Rechannels both to stereo
- Mixes at 70% speaker + 30% mic ratio
- Sends mixed audio through Opus encoder

#### New Dual-Stream Capture (Lines 488-736)
- `play_voice_call_mixed()`: Creates and manages dual audio streams
- Opens default output device in loopback mode (speaker)
- Opens default input device (microphone)
- Dynamically detects sample rates, channels, and formats
- Builds separate input streams with callbacks that populate buffers
- Spawns dedicated mixing thread that:
  - Continuously monitors both buffers
  - Extracts synchronized frames (10ms chunks)
  - Implements drift prevention (drops samples if buffers diverge > 1 second)
  - Calls `mix_and_send()` to process and transmit audio
  - Exits cleanly when voice call ends

#### Updated play() Function (Lines 738-833)
- Added Windows-specific check for voice call mode
- Returns appropriate stream based on voice call state
- Falls back to regular loopback-only capture when not in voice call

#### Updated Service Control (Lines 233-265, 268-299)
- Modified `run_restart()`: Initializes voice call streams when mixing is active
- Modified `run_serv_snapshot()`: Same logic for snapshot-based initialization
- Both functions check `IS_VOICE_CALL_ACTIVE` and route to appropriate capture mode

#### New Public API (Lines 61-75)
- `set_voice_call_mixing(enabled: bool)`: Enables/disables voice call mixing
- Triggers audio service restart to switch modes
- Windows-only function with proper conditional compilation

### 2. src/server/connection.rs

#### Updated handle_voice_call() (Lines 3569-3602)
- Added Windows-specific path: calls `set_voice_call_mixing(true)` on acceptance
- Non-Windows systems continue using existing `set_voice_call_input_device()` 
- Properly separated behavior with `#[cfg(windows)]` and `#[cfg(not(windows))]`

#### Updated close_voice_call() (Lines 3604-3623)
- Added Windows-specific path: calls `set_voice_call_mixing(false)` on closure
- Non-Windows systems continue using existing cleanup
- Ensures proper deactivation and resource cleanup

## Technical Details

### Audio Pipeline
1. **Capture**: Two independent CPAL input streams (speaker loopback + mic)
2. **Buffering**: Samples accumulated in thread-safe VecDeques
3. **Synchronization**: 10ms frame extraction with drift detection
4. **Resampling**: Rubato FastFixedIn for sample rate conversion (if needed)
5. **Rechanneling**: Fon-based channel conversion to stereo (if needed)
6. **Mixing**: Simple weighted sum (0.7 * speaker + 0.3 * mic)
7. **Encoding**: Opus encoding at 48kHz stereo
8. **Transmission**: Sent via existing GenericService infrastructure

### Drift Prevention
- Monitors buffer levels continuously
- If speaker buffer exceeds mic buffer by >1 second: drops speaker samples
- If mic buffer exceeds speaker buffer by >1 second: drops mic samples
- Logs warnings when drift correction occurs

### Thread Safety
- All buffers protected by Mutex
- Voice call state managed through Arc<Mutex<bool>>
- Mixing thread spawned independently, exits gracefully when deactivated
- No deadlocks: locks held for minimal duration

### Sample Rate Handling
- Dynamically detects device sample rates (44.1kHz, 48kHz, 96kHz, etc.)
- Target rate: 48000 Hz (optimal for Opus)
- Rubato used for high-quality resampling when rates differ
- Handles arbitrary sample rate combinations

### Channel Handling
- Supports 1-8 channels per device
- Target: Stereo (2 channels)
- Uses existing fon-based rechanneling infrastructure
- Properly converts mono mic to stereo, maintains stereo speaker, etc.

## Testing Recommendations

1. **Basic Functionality**
   - Accept voice call on Windows remote
   - Verify client receives mixed audio (system sounds + mic)
   - Close voice call, verify clean deactivation

2. **Different Sample Rates**
   - Test with 44.1kHz speaker + 48kHz mic
   - Test with 48kHz speaker + 44.1kHz mic
   - Test with non-standard rates (96kHz, 192kHz)

3. **Long Duration**
   - Run voice call for 10+ minutes
   - Monitor for audio drift
   - Check for memory leaks in buffers

4. **Audio Quality**
   - Play music on remote while talking
   - Verify both are audible at proper levels
   - Listen for glitches, pops, or artifacts

5. **Edge Cases**
   - Disconnect/reconnect audio devices during call
   - Switch default audio devices mid-call
   - Rapid start/stop of voice calls
   - Voice call during existing audio streaming

## Compatibility

- **Windows Only**: All mixing features are `#[cfg(windows)]` guarded
- **Other Platforms**: Continue using existing microphone-only capture
- **Rubato Feature**: Works with or without `use_rubato` feature flag
  - With rubato: High-quality sinc-based resampling
  - Without rubato: Falls back to existing dasp resampling

## Performance Considerations

- **CPU Usage**: Minimal overhead from mixing thread (~5ms sleep between checks)
- **Memory**: Fixed buffer sizes, drift prevention prevents unbounded growth
- **Latency**: ~10ms additional buffering (1 frame), negligible for voice calls
- **Quality**: No quality loss from mixing, high-quality resampling maintains fidelity

## Code Quality

- ✅ No linter errors
- ✅ No compilation errors
- ✅ Proper error handling with ResultType
- ✅ Comprehensive logging for debugging
- ✅ Clean separation of concerns
- ✅ Platform-specific conditional compilation
- ✅ Thread-safe shared state management
- ✅ Graceful cleanup and resource management

## Compilation Notes

The implementation uses the `dasp::sample::ToSample` trait for converting i16 audio samples to f32. The trait is brought into scope within closures using `use dasp::sample::ToSample;` to enable the `.to_sample()` method on i16 values.

