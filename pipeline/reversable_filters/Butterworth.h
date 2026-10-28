#ifndef BUTTERWORTH
#define BUTTERWORTH
#include <stdio.h>
#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    char chunkId[4];        // "RIFF"
    uint32_t chunkSize;
    char format[4];         // "WAVE"
    char subchunk1Id[4];    // "fmt "
    uint32_t subchunk1Size; // 16 for PCM
    uint16_t audioFormat;   // 1 = PCM
    uint16_t numChannels;   // 1 = Mono, 2 = Stereo
    uint32_t sampleRate;    // e.g., 44100
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample; // e.g., 16
    char subchunk2Id[4];    // "data"
    uint32_t subchunk2Size;
} WavHeader;
#pragma pack(pop)

void apply_bilinear_lowpass(const char* input_filename, const char* output_filename, double cutoff_freq_hz, int order);

void apply_bilinear_highpass(const char* input_filename, const char* output_filename, double cutoff_freq_hz, int order);

void reverse_bilinear_lowpass(const char* input_filename, const char* output_filename, double cutoff_freq_hz, int order);

void reverse_bilinear_bandpass(const char *input_filename, const char *output_filename,
double hp_cutoff_freq_hz, double lp_cutoff_freq_hz, int order);

#endif
