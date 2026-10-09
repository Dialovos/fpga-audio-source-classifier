# Project Plan - CSC4240

The report and code are due Wed Nov 25, so the last stage ends Wed Nov 18. That leaves a full week for final adjustments.

We start today, Fri Oct 9. Vadin starts Oct 12 and Adam starts Oct 14, so Stage 1 is the longest stage to give everyone time to get going.

For the progress check-in (Oct 25 - Nov 5), we should book a late slot (~Nov 2-5) so we have Stage 1 and early Stage 2 results to show.

| Stage | Dates | Deadline |
| --- | --- | --- |
| 1. Test data, filter ID, recovery limit | Oct 9 - Oct 25 | Sun Oct 25 |
| 2. Recovery pipeline & event detection | Oct 26 - Nov 8 | Sun Nov 8 |
| 3. MicroBlaze benchmark, final results, report | Nov 9 - Nov 18 | Wed Nov 18 |
| Buffer | Nov 19 - Nov 25 | Wed Nov 25 (turn-in) |

Team meeting: weekend of Oct 17-18 to start tying things together. The 17th is a Saturday, so we still need to pick the day.

## Roles

Based on our first meeting, adjusted for the revised proposal.

| Person | Area | What they own now |
| --- | --- | --- |
| Adam | FPGA / hardware (SystemVerilog/VHDL, Vivado) | Test data method for the third revision, the MicroBlaze hardware platform & `.xsa`, and FPGA acceleration if we have time |
| Le | ML & repo (Python) | Test signal generator code, AI layer 1 (filter ID), AI layer 2 (event classifier), exporting weights, and the repo & PRs |
| Vadin | | |
| John | Audio / DSP (C, Python first) | Cleaning the event sounds, the inverse filter, the recovery limit test, and checking the MicroBlaze output matches Python |
| Tomu | | |

- Vadin and Tomu pick from the tasks marked Open below.

- Everyone writes up their own part for the report, since the course wants a section on what each person did.

## Stage 1 - Test Data, Filter ID, and the Recovery Limit (due Sun Oct 25)

Goal: show the first AI layer works, and find out how much filtered out sound we can get back.

- Adam: write up the test data method for the third revision ASAP. Le builds the generator code from it, with a first version ready for the meeting.

- Le: Build the test signal generator. Mix speech with event sounds from public datasets, plus cabin noise.

- Le: Run every mix through a random high-pass, low-pass, or band-pass filter with a known type, order, and cutoff. We make the filters ourselves, so the labels come for free.

- Le: AI layer 1: predict the filter type and order, estimate the cutoffs, from the filtered audio. Only try a small neural net if that falls short.

- John: pick and clean the event sounds the generator uses.

- John: Recovery limit test: apply the inverse filter with a cap on how much it can boost, then measure how much of a buried event comes back as the cut gets deeper. Anything pushed below the noise floor is gone for good, so this test decides what we can claim.

- Le & John: Check: blind filter identification, forensic audio restoration, published acoustic analysis of cockpit voice recorders.

- Open: Hardware in parallel: get a basic program running on the MicroBlaze and measure cycle counts, so Stage 3 doesn't start from zero.

- Adam (from Oct 14): build the MicroBlaze hardware platform and export the `.xsa` for the MicroBlaze software.

- Open: sketch the flow on the board (what runs in what order, and what goes out over UART) and start the PC side that reads it.

Done when: we have the filter ID accuracy, a plot of the recovery limit, and the generator script in the repo.

## Stage 2 - Recovery Pipeline and Event Detection (due Sun Nov 8)

Goal: show that recovery helps event detection, start to finish.

- John: Inverse filter using the estimated filter. Also run it with the true filter so we know the best case.

- Le: AI layer 2: an event classifier built on a pretrained audio model with a small classifier on top.

- Le: Main comparison: detect events on the filtered audio as-is, after recovery with the estimated filter, after recovery with the true filter, and on the original audio as the ceiling.

- Le & Open: Get the small models ready for the board: shrink them and port them to C.

- Adam: FPGA acceleration for the inverse filter or the classifier, only if the basics are on track.

- Everyone: Check-in (late slot, ~Nov 2-5): present the Stage 1 results and the early Stage 2 results.

Done when: we have a table and figures for the main comparison.

## Stage 3 - MicroBlaze Benchmark, Final Results, and Report (due Wed Nov 18)

- Open: Run the whole pipeline on the MicroBlaze with recorded test signals: filter ID, the inverse filter, event classifier. Measure speed and memory, and check the outputs match the PC version.

- Le: Final runs: repeat the experiments on held-out sounds and filters, and finish the figures.

- Everyone: Write the 4-page report draft: method, ideas we tried and dropped, implementation, eval, and what each person did.

- Le: Code: a README and notebooks that run from a clean setup, and market analysis is for the bonus.

Done when: we have a full report draft and code anyone can run.

## Buffer - Final Adjustments (Nov 19 - Nov 25)

- Fixes, reruns, proofreading, and turn-in on Wed Nov 25.

- Everyone: Build the slides for the presentation the week of Dec 1.

## Things to Keep in Mind

- Stage 2 sticks to detecting and classifying events.

- CVR recordings aren't available. Under 49 U.S.C. 1114(c), the NTSB "may not disclose publicly any part of a cockpit voice or video recorder recording." US, the NTSB can release transcripts but not the recordings, and EU rules keep recordings for safety investigations only. A few recordings have leaked over the years, but there's no dataset of them and none come with known filter settings, so we use simulated test signals instead.
