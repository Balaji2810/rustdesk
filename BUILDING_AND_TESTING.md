# Building and Testing Windows Voice Call Audio Mixing

## Building

### Prerequisites
- Rust toolchain (1.75+)
- Windows SDK
- VCPKG dependencies (as per existing RustDesk requirements)

### Build Commands

#### Development Build
```bash
cargo build --features use_dasp
```

#### With Rubato (Recommended for Production)
```bash
cargo build --release --features use_rubato
```

#### Full Feature Set
```bash
cargo build --release --features "use_rubato,hwcodec"
```

## Testing the Feature

### Setup
1. Build RustDesk with the changes on a Windows machine (remote side)
2. Install/run the build on the Windows machine
3. Connect to it from another RustDesk client (controller)

### Test Procedure

#### Test 1: Basic Voice Call with Mixing
1. From controller, initiate a voice call to the Windows remote
2. On remote, accept the voice call
3. **Expected**: You should hear both:
   - Any audio playing on the remote system (YouTube, Spotify, etc.)
   - The remote user speaking into their microphone
4. Verify the mix ratio sounds appropriate (speaker slightly louder than mic)

#### Test 2: Verify Logs
Check the remote system logs for:
```
Voice call mixing: true
Starting voice call with mixed audio (speaker + mic)
Voice call speaker device: [Device Name]
Voice call speaker format: [Format Details]
Voice call mic device: [Device Name]
Voice call mic format: [Format Details]
```

#### Test 3: Voice Call Closure
1. End the voice call from either side
2. **Expected**: Clean deactivation
3. Check logs for:
```
Voice call mixing: false
Voice call mixing thread exiting
```

#### Test 4: Sample Rate Compatibility
1. Test with different audio devices as defaults:
   - USB headset (often 44.1kHz)
   - Built-in audio (usually 48kHz)
   - Studio equipment (may be 96kHz or 192kHz)
2. **Expected**: No audio artifacts regardless of device sample rates

#### Test 5: Long Duration
1. Start a voice call
2. Play continuous audio on the remote (e.g., music)
3. Let it run for 10-15 minutes
4. **Expected**: No audio drift, glitches, or degradation over time
5. Check for drift prevention warnings in logs (should be none under normal conditions)

#### Test 6: Audio Mix Ratio
1. Start voice call
2. Play known audio on remote at consistent volume
3. Speak into remote microphone at consistent volume
4. **Expected**: 
   - System audio should be noticeably louder (70%)
   - Microphone should be quieter but still clear (30%)
   - Both should be audible and intelligible

#### Test 7: Platform Isolation
1. Test voice call on Linux remote
2. **Expected**: Uses microphone only (existing behavior)
3. Test voice call on macOS remote  
4. **Expected**: Uses microphone only (existing behavior)

### Debugging

#### Enable Verbose Logging
Set environment variable before running:
```bash
set RUST_LOG=debug
rustdesk.exe
```

#### Common Issues

**Issue**: No speaker audio in mix, only microphone
- Check: Is audio actually playing on the remote?
- Check: Log shows "Voice call speaker device: [name]"?
- Check: Windows audio output device is set correctly

**Issue**: Audio is choppy or has artifacts
- Check logs for: "Speaker buffer too far ahead" or "Mic buffer too far ahead"
- This indicates drift - may need sample rate adjustment
- Try different default audio devices

**Issue**: No microphone audio in mix
- Check: Microphone is set as default input device in Windows
- Check: Microphone permissions are granted
- Check: Log shows "Voice call mic device: [name]"?

**Issue**: Compilation errors on non-Windows platforms
- Ensure all Windows-specific code is properly guarded with `#[cfg(windows)]`
- This should not happen with current implementation

**Issue**: High CPU usage
- Check for excessive drift correction in logs
- May indicate clock sync issues between audio devices
- Try using USB audio devices instead of different interfaces

### Performance Monitoring

Monitor these metrics during testing:

1. **CPU Usage**: Should be <5% additional during voice call
2. **Memory Usage**: Should be stable, not growing
3. **Audio Latency**: Should be <50ms end-to-end
4. **Buffer Levels**: Check logs for buffer overrun warnings

### Code Quality Checks

Before deploying:
```bash
# Check for compilation errors
cargo check --all-features

# Run clippy for warnings
cargo clippy --all-features

# Check formatting
cargo fmt --check

# Run tests (if any exist)
cargo test --all-features
```

## Rollback

If issues are found, voice call mixing can be disabled by:

1. **Code Change**: Comment out the mixing activation in `connection.rs`:
```rust
#[cfg(windows)]
{
    // crate::audio_service::set_voice_call_mixing(true);
    crate::audio_service::set_voice_call_input_device(
        crate::get_default_sound_input(),
        false,
    );
}
```

2. **Runtime**: The feature is per-voice-call, so it resets each time

## Notes

- Feature is Windows-only by design
- No configuration UI changes needed - activates automatically
- Compatible with existing RustDesk infrastructure
- No breaking changes to existing functionality

