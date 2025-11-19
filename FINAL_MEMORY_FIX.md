# Final Memory Leak Fix: Periodic Segmenter Recreation

## The Root Cause

**MediaPipe's C++ code holds memory that Python's garbage collector CANNOT free.**

Even though:
- ✅ We limit the async queue
- ✅ We copy arrays to break references  
- ✅ We call `gc.collect()` frequently
- ✅ We delete all Python references

**MediaPipe's internal C++ buffers still accumulate memory** because Python GC has no control over C++ memory.

## The Solution: Periodic Segmenter Recreation

Every **500 frames** (~50 seconds at 10 FPS), the code now:

1. **Waits for pending operations** to complete
2. **Closes the old segmenter** (releases C++ resources)
3. **Forces Python GC** to collect Python-side objects
4. **Creates a new segmenter** (fresh C++ memory)

This **forces MediaPipe to release all its internal C++ buffers**.

## What You'll See

Every ~50 seconds, you'll see:
```
🔄 Recreating segmenter to force memory release (processed 500 frames)...
✅ Segmenter recreated
```

This happens automatically and should **prevent memory from growing continuously**.

## Performance Impact

- **Brief pause** (~100-200ms) during recreation
- **No visual interruption** - happens in background
- **Memory should stabilize** instead of growing

## Configuration

You can adjust the frequency by changing:
```python
self._max_segmenter_frames = 500  # Change this value
```

- **Lower value** (e.g., 300) = More frequent recreation, better memory control, more pauses
- **Higher value** (e.g., 1000) = Less frequent recreation, less pauses, more memory growth

## Expected Results

- **Memory should stabilize** after each recreation
- **No continuous growth** over long sessions
- **Periodic resets** keep memory under control

## Why This Works

When you call `segmenter.close()`, MediaPipe's C++ destructor runs, which:
- Releases GPU memory buffers
- Frees internal queues
- Clears cached data
- Resets all internal state

Creating a new segmenter starts fresh with clean memory.

## Alternative Solutions (Not Implemented)

1. **Synchronous Processing**: Would prevent queue buildup but hurt performance significantly
2. **Lower-level MediaPipe API**: Would require rewriting significant code
3. **Memory-mapped files**: Complex and may not help with GPU memory

**Periodic recreation is the most practical solution** that balances performance and memory control.

