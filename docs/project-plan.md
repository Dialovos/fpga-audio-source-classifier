# Project Plan - CSC4240

TThe report and code are due Wed Nov 25, so the last stage ends Wed Nov 18. That leaves a full week for final adjustments.

Stage 1 ends before the progress check-in window (Oct 25 - Nov 5) so we have results to show. Deadlines land on Sundays to line up with our meetings, except Stage 3, which ends on the buffer date.

| Stage | Dates | Deadline |
| --- | --- | --- |
| 1. Data, filter ID, recovery limit | Oct 6 - Oct 18 | Sun Oct 18 |
| 2. Recovery pipeline & event detection | Oct 19 - Nov 8 | Sun Nov 8 |
| 3. MicroBlaze benchmark, final results, report | Nov 9 - Nov 18 | Wed Nov 18 |
| Buffer | Nov 19 - Nov 25 | Wed Nov 25 (turn-in) |

## Stage 1 - Data, Filter ID, and the Recovery Limit (due Sun Oct 18)

Goal: show the first AI layer works, and find out how much filtered out sound we can get back.

- Build the test signal generator. Mix speech with event sounds from public datasets, plus cabin noise.

- Run every mix through a random high-pass, low-pass, or band-pass filter with a known type, order, and cutoff. We make the filters ourselves, so the labels come for free.

- AI layer 1: predict the filter type and order, estimate the cutoffs, from the filtered audio. Only try a small neural net if that falls short.

- Recovery limit test: apply the inverse filter with a cap on how much it can boost, then measure how much of a buried event comes back as the cut gets deeper. Anything pushed below the noise floor is gone for good, so this test decides what we can claim.

- Check: blind filter identification, forensic audio restoration, published acoustic analysis of cockpit voice recorders.

- Hardware in parallel: get a basic program running on the MicroBlaze and measure cycle counts, so Stage 3 doesn't start from zero.

Done when: we have the filter ID accuracy, a plot of the recovery limit, and the generator script in the repo.

## Stage 2 - Recovery Pipeline and Event Detection (due Sun Nov 8)

Goal: show that recovery helps event detection, start to finish.

- Inverse filter using the estimated filter. Also run it with the true filter so we know the best case.

- AI layer 2: an event classifier built on a pretrained audio model with a small classifier on top.

- Main comparison: detect events on the filtered audio as-is, after recovery with the estimated filter, after recovery with the true filter, and on the original audio as the ceiling.

- Get the small models ready for the board: shrink them and port them to C.

- Check-in: present the Stage 1 results and the early Stage 2 results.

Done when: we have a table and figures for the main comparison.

## Stage 3 - MicroBlaze Benchmark, Final Results, and Report (due Wed Nov 18)

- Run the whole pipeline on the MicroBlaze with recorded test signals: filter ID, the inverse filter, event classifier. Measure speed and memory, and check the outputs match the PC version.

- Final runs: repeat the experiments on held-out sounds and filters, and finish the figures.

- Write the 4-page report draft: method, ideas we tried and dropped, implementation, eval, and what each person did.

- Code: a README and notebooks that run from a clean setup, and market analysis is for the bonus.

Done when: we have a full report draft and code anyone can run.

## Buffer - Final Adjustments (Nov 19 - Nov 25)

- Fixes, reruns, proofreading, and turn-in on Wed Nov 25.

- Build the slides for the presentation the week of Dec 1.

## Things to Keep in Mind

- Stage 2 sticks to detecting and classifying events.

- CVR recordings aren't available. Under 49 U.S.C. 1114(c), the NTSB "may not disclose publicly any part of a cockpit voice or video recorder recording." US, the NTSB can release transcripts but not the recordings, and EU rules keep recordings for safety investigations only. A few recordings have leaked over the years, but there's no dataset of them and none come with known filter settings, so we use simulated test signals instead.
