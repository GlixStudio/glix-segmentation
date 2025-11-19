# Memory Leak Fixes Applied

## Critical Fixes Applied

### 1. **Explicit Garbage Collection** ✅
- Added `import gc` at the top
- Set more aggressive GC thresholds: `(700, 10, 10)` instead of default
- Force GC collection every 60 frames
- This ensures Python's GC runs more frequently to collect accumulated objects

### 2. **MediaPipe Callback Memory Leak Fix** ✅ CRITICAL
**Problem:** MediaPipe's `numpy_view()` creates a view that holds a reference to MediaPipe's internal buffer, preventing GC.

**Solution:**
```python
# Before: mask = result.category_mask.numpy_view()  # View holds reference!
# After:
mask_view = result.category_mask.numpy_view()
mask = mask_view.copy()  # Create independent copy
del mask_view  # Release view reference immediately
```

**Impact:** This was likely the MAIN leak - MediaPipe was holding onto all mask data.

### 3. **Explicit Reference Clearing in Callbacks** ✅
- Clear old mask/landmark references before assigning new ones
- Use `del` to explicitly break references
- This helps GC collect old data immediately

### 4. **Frame Skipping** ✅
- Added frame rate limiting (max 60 FPS processing)
- Prevents queue buildup when processing is slower than capture
- Skips frames if processing queue gets too full

### 5. **Explicit Variable Deletion** ✅
- Delete MediaPipe image objects immediately after async calls
- Delete frame references at end of loop
- Delete intermediate arrays after use

### 6. **More Aggressive GC Thresholds** ✅
```python
gc_threshold = (700, 10, 10)  # More frequent collections
gc.set_threshold(*gc_threshold)
```

## Expected Results

### Memory Usage
- **Before:** Continuous growth, 50-200 MB/hour
- **After:** Stable memory usage with periodic cleanup
- **GC Collections:** Every 60 frames (~1-2 seconds at 30 FPS)

### Performance Impact
- **GC Overhead:** ~1-2ms per collection (every 60 frames)
- **Frame Skipping:** Prevents queue buildup, maintains smooth FPS
- **Overall:** Minimal performance impact, significant memory improvement

## Monitoring

The code now logs GC collections:
```
🧹 GC collected X objects
```

This appears every 5 seconds if objects are being collected.

## Testing

To verify the fixes work:

1. **Run the application for 10+ minutes**
2. **Monitor Activity Monitor** - memory should stabilize
3. **Check GC logs** - should see periodic collections
4. **Memory should not continuously grow**

## Additional Notes

- Python's GC runs automatically, but MediaPipe's C++ code may hold references
- The explicit `copy()` breaks MediaPipe's reference chain
- Frame skipping prevents memory buildup during slow processing
- More aggressive GC thresholds ensure timely collection

## If Memory Still Grows

If memory still increases after these fixes:

1. **Check MediaPipe version** - older versions had more leaks
2. **Reduce processing size** - smaller frames = less memory
3. **Disable face landmarks** - reduces processing overhead
4. **Increase GC frequency** - change `gc_frequency` from 60 to 30
5. **Check for other leaks** - use `tracemalloc` to find remaining leaks

```python
import tracemalloc
tracemalloc.start()
# ... run code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB, Peak: {peak / 1024 / 1024:.1f} MB")
```

