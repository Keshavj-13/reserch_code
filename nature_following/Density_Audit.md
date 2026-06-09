# DENSITY AUDIT

## PROBLEM: Dense Paragraphs
**Location:** Methods - Environment (Lines 167-183)
**Analysis:** A single 16-line paragraph describing simulator physics, training episodes, and termination.
**Fix:** Split into two: (1) Thermal Dynamics & Initialization, (2) Termination & Safety Logic.

**Location:** Methods - Inputs (Lines 196-210)
**Analysis:** Explains global, temporal, and zone features in one block.
**Fix:** Use a bulleted list for the three streams.

## PROBLEM: Equation Clumping
**Location:** Predictive Encoders (Lines 468-490)
**Analysis:** Seven equations in a row (H1, H2, H3, zg, zt, zs, z).
**Fix:** Interleave equations with descriptive text explaining the *role* of each step (e.g., "The node embeddings are iteratively refined through three hops...").

## PROBLEM: Hard to Scan Pages
**Location:** Page 3/4 (Methods section middle)
**Analysis:** Almost no paragraph breaks for several columns.
**Fix:** Add sub-sub-headings or bolded keywords at the start of paragraphs (e.g., "**Spatial Encoding.**", "**Temporal Memory.**").

## PROBLEM: Abstract Visual Weight
**Analysis:** The abstract is one large block.
**Fix:** Some journals allow (or prefer) structured abstracts. Even if not, ensuring the sentences aren't too long helps. The current abstract is 12 lines of text. This is borderline okay but could be lightened by removing the exhaustive list of baselines.
