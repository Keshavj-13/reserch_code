# Telemetry Budget Calculation

## Assumptions
- Data type: Float32 (4 bytes per value)
- Total steps (horizon): 6105
- Epochs: 50
- Controllers: 4
- Decimation Factor (N): 20

## Full Rate Tensors (Saved every step)
- Rewards: 1 value
- Temperatures: 12 values
- Actions: 14 values
- **Total:** 27 values/step = 108 bytes/step
- **Per episode:** 6105 * 108 = 659,340 bytes (~0.66 MB)

## Decimated Tensors (Saved every N=20 steps)
- Latents (Fused): 448 values
- GraphSAGE Embeddings: 128 values
- LSTM Embeddings: 256 values
- Actor outputs: 64 values
- Critic value: 1 value
- **Total:** 897 values/step = 3588 bytes/step
- **Steps saved:** 6105 / 20 = 305 steps
- **Per episode:** 305 * 3588 = 1,094,340 bytes (~1.09 MB)

## Total Calculations
1. **Per Controller Run (1 episode):** 0.66 MB + 1.09 MB = 1.75 MB
2. **Total Campaign (4 controllers * 50 epochs):** 200 runs
3. **Campaign Size:** 200 * 1.75 MB = 350 MB (0.35 GB)

## Conclusion
The volume is verified safe for Windows NTFS. The calculation proves that decimating heavy tensors completely avoids the gigabyte-scale write bursts that caused the Linux BTRFS failure.
