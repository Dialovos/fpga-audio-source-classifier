#include "Butterworth.h"
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define M_PI 3.14159265358979323846
#define M_SQRT2 1.41421356237309504880

const double BUTTERWORTH_Q_LOOKUP[16][8] = {
      {0.0}, // n = 0 (Unused)
      {0.0}, // n = 1 (Single real pole, no quadratic section)

      // n = 2
      {7.07106781186547524e-01},

      // n = 3 (Plus 1 real pole)
      {1.00000000000000000e+00},

      // n = 4
      {1.30656296487637658e+00, 5.41196100146197012e-01},

      // n = 5 (Plus 1 real pole)
      {1.61803398874989490e+00, 6.18033988749894903e-01},

      // n = 6
      {1.93185165257813751e+00, 7.07106781186547524e-01,
       5.17638090205041479e-01},

      // n = 7 (Plus 1 real pole)
      {2.24697960371746763e+00, 8.01937735804838292e-01,
       5.54958132087371214e-01},

      // n = 8
      {2.56291544774150726e+00, 8.99976223136416120e-01,
       6.01344886935045175e-01, 5.09795579104159180e-01},

      // n = 9 (Plus 1 real pole)
      {2.87938524157181730e+00, 1.00000000000000000e+00,
       6.52703644666139393e-01, 5.32088886237956138e-01},

      // n = 10
      {3.19622661074983183e+00, 1.10134463229263346e+00,
       7.07106781186547524e-01, 5.61163118817180395e-01,
       5.06232562894001537e-01},

      // n = 11 (Plus 1 real pole)
      {3.51333709166613861e+00, 1.20361562377556530e+00,
       7.63521118433367674e-01, 5.94351144437140433e-01,
       5.21108558113202780e-01},

      // n = 12
      {3.83064878777019402e+00, 1.30656296487637658e+00,
       8.21339815852290767e-01, 6.30236207005132276e-01,
       5.41196100146197012e-01, 5.04314480290076417e-01},

      // n = 13 (Plus 1 real pole)
      {4.14811490527938354e+00, 1.41002004842652906e+00,
       8.80181357630755667e-01, 6.67993079887886698e-01,
       5.64680780879146416e-01, 5.14963915474863754e-01},

      // n = 14
      {4.46570213519025572e+00, 1.51387132154284543e+00,
       9.39792959993807631e-01, 7.07106781186547524e-01,
       5.90511054787033030e-01, 5.29726486256145157e-01,
       5.03163788290089009e-01},

      // n = 15 (Plus 1 real pole)
      {4.78338611675281911e+00, 1.61803398874989490e+00,
       1.00000000000000000e+00, 7.47238274932304325e-01,
       6.18033988749894903e-01, 5.47318139253023417e-01,
       5.11170297432514631e-01}};

void apply_bilinear_lowpass(const char *input_filename,
                            const char *output_filename,
                            double cutoff_freq_hz, int order) {
  FILE *infile = fopen(input_filename, "rb");
  if (!infile) {
    perror("Failed to open input WAV");
    return;
  }

  WavHeader header;
  
  if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
    perror("Failed to read WAV header");
    fclose(infile);
    return;
}
  fseek(infile, 12, SEEK_SET);

  char chunk_id[4];
  uint32_t chunk_size;
  int found_data = 0;

  while (fread(chunk_id, 1, 4, infile) == 4) {

      if (fread(&chunk_size, sizeof(uint32_t), 1, infile) != 1)
        break;

      if (memcmp(chunk_id, "data", 4) == 0) {
        found_data = 1;
        printf("Found data chunk: %u bytes\n", chunk_size);

         break;
      }

      // Skip this chunk, including optional RIFF padding
      fseek(infile, chunk_size + (chunk_size & 1u), SEEK_CUR);
  }
  
  if (!found_data) {
    fprintf(stderr, "WAV data chunk not found\n");
    fclose(infile);
    return;
  } 
  
  // Save the audio starting position
  long audio_offset = ftell(infile);

  
  
  printf("RIFF ID: %.4s\n", header.chunkId);
  printf("WAVE ID: %.4s\n", header.format);

  printf("Format chunk: %.4s\n", header.subchunk1Id);
  printf("Format size: %u\n", header.subchunk1Size);

  printf("Audio format: %u\n", header.audioFormat);
  printf("Channels: %u\n", header.numChannels);
  printf("Sample rate: %u\n", header.sampleRate);
  printf("Bit depth: %u\n", header.bitsPerSample);

  printf("Data chunk: %.4s\n", header.subchunk2Id);
  printf("Data size: %u\n", header.subchunk2Size);

 
  // Basic check  mono/stereo PCM
  double fs = (double)header.sampleRate;
  double T = 1.0 / fs;

  // Pre-warp cutoff frequency
  double omega_c = 2.0 * M_PI * cutoff_freq_hz;
  double Omega_c = (2.0 / T) * tan(omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double Q_multiplier = 1.0;
  double k = Omega_c * T / 2.0;

  double gamma, b0, b1, b2, a1, a2;
  double gamma1 = 0, b0_1 = 0, b1_1 = 0;
  double b2_1 = 0, a1_1 = 0, a2_1 = 0;

  if (order == 1) {

      gamma = 1.0 + k;

      b0 = k / gamma;
      b1 = k / gamma;
      b2 = 0.0;

      a1 = (k - 1.0) / gamma;
      a2 = 0.0;

  } else if (order == 2 || order == 4) {

      double Q = BUTTERWORTH_Q_LOOKUP[order][0] / Q_multiplier;

      gamma = 1.0 + k / Q + k*k;

      b0 = k*k / gamma;
      b1 = 2.0*k*k / gamma;
      b2 = k*k / gamma;

      a1 = 2.0*(k*k - 1.0) / gamma;
      a2 = (1.0 - k/Q + k*k) / gamma;

      if (order == 4) {

          double Q1 = BUTTERWORTH_Q_LOOKUP[4][1] / Q_multiplier;

          gamma1 = 1.0 + k/Q1 + k*k;

          b0_1 = k*k / gamma1;
          b1_1 = 2.0*k*k / gamma1;
          b2_1 = k*k / gamma1;

          a1_1 = 2.0*(k*k - 1.0) / gamma1;
          a2_1 = (1.0 - k/Q1 + k*k) / gamma1;
      }
  }

  FILE *outfile = fopen(output_filename, "wb");
  if (!outfile) {
    fclose(infile);
    perror("Failed to open output WAV");
    return;
  }
  
  // Return to beginning of input WAV
  fseek(infile, 0, SEEK_SET);

  // Copy complete WAV header
  for (long i = 0; i < audio_offset; i++) {

      int byte = fgetc(infile);

      if (byte == EOF)
          break;

      fputc(byte, outfile);
  }

  // Process 32-bit samples
  size_t samples_to_read = chunk_size / sizeof(float);
  
  printf("Calculated samples: %zu\n", samples_to_read);


  double x_prev1 = 0.0, y_prev1 = 0.0;
  double x_prev2 = 0.0, y_prev2 = 0.0;

  double x1_prev1 = 0.0, y1_prev1 = 0.0;
  double x1_prev2 = 0.0, y1_prev2 = 0.0;

  float sample_in;

  for (size_t i = 0; i < samples_to_read; i++) {

    if (fread(&sample_in, sizeof(float), 1, infile) != 1)
      break;

    double x = (double)sample_in;

    // First low-pass stage
    double y1 =
        b0 * x + b1 * x_prev1 + b2 * x_prev2 - a1 * y_prev1 - a2 * y_prev2;

    x_prev2 = x_prev1;
    x_prev1 = x;
    y_prev2 = y_prev1;
    y_prev1 = y1;
    // Second low-pass stage
    double y2;

    if (order == 4) {
        y2 = b0_1*y1 + b1_1*x1_prev1 + b2_1*x1_prev2 - a1_1*y1_prev1 - a2_1*y1_prev2;

        x1_prev2 = x1_prev1;
        x1_prev1 = y1;

        y1_prev2 = y1_prev1;
        y1_prev1 = y2;

    } else {
        y2 = y1;
    }

    float sample_out = (float)y2;

    fwrite(&sample_out, sizeof(float), 1, outfile);
  }

  fclose(infile);
  fclose(outfile);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void apply_bilinear_highpass(const char *input_filename,
                             const char *output_filename,
                             double cutoff_freq_hz,int order) {
  FILE *infile = fopen(input_filename, "rb");
  if (!infile) {
    perror("Failed to open input WAV");
    return;
  }

  WavHeader header;
  
  if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
    perror("Failed to read WAV header");
    fclose(infile);
    return;
}
  fseek(infile, 12, SEEK_SET);

  char chunk_id[4];
  uint32_t chunk_size;
  int found_data = 0;

  while (fread(chunk_id, 1, 4, infile) == 4) {

      if (fread(&chunk_size, sizeof(uint32_t), 1, infile) != 1)
        break;

      if (memcmp(chunk_id, "data", 4) == 0) {
        found_data = 1;
        printf("Found data chunk: %u bytes\n", chunk_size);

         break;
      }

      // Skip this chunk, including optional RIFF padding
      fseek(infile, chunk_size + (chunk_size & 1u), SEEK_CUR);
  }
  
  if (!found_data) {
    fprintf(stderr, "WAV data chunk not found\n");
    fclose(infile);
    return;
  } 
  
  // Save the audio starting position
  long audio_offset = ftell(infile);

  // Basic check  mono/stereo PCM
  double fs = (double)header.sampleRate;
  double T = 1.0 / fs;

  // Pre-warp cutoff frequency
  double omega_c = 2.0 * M_PI * cutoff_freq_hz;
  double Omega_c = (2.0 / T) * tan(omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double Q_multiplier = 1.0;

  double k = Omega_c * T / 2.0;
  
  double gamma, b0, b1, b2, a1, a2;
  double gamma1 = 0, b0_1 = 0, b1_1 = 0;
  double b2_1 = 0, a1_1 = 0, a2_1 = 0;
  if (order == 1) {

      	gamma = 1.0 + k;

	b0 = 1.0 / gamma;
	b1 = -1.0 / gamma;
	b2 = 0.0;

	a1 = (k - 1.0) / gamma;
	a2 = 0.0;

  } else if (order == 2 || order == 4) {

        gamma = 1.0 + (Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * k + k*k;

        b0 = 1 / gamma;
        b1 = -2 / gamma;
        b2 = 1 / gamma;

        a1 = 2 * (pow(k, 2) - 1.0) / gamma;
        a2 = (1.0 - (Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * k + k*k) / gamma;

      if (order == 4) {

          gamma1 = 1.0 + (Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1]) * k + pow(k, 2);

          b0_1 = 1 / gamma1;
          b1_1 = -2 / gamma1;
          b2_1 = 1 / gamma1;

          a1_1 = 2 * (pow(k, 2) - 1.0) / gamma1;
          a2_1 = (1.0 - Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1] * k + pow(k, 2)) /
      gamma1;
      }
  }
  FILE *outfile = fopen(output_filename, "wb");
  if (!outfile) {
    fclose(infile);
    perror("Failed to open output WAV");
    return;
  }
  
  // Return to beginning of input WAV
  fseek(infile, 0, SEEK_SET);

  // Copy complete WAV header
  for (long i = 0; i < audio_offset; i++) {

      int byte = fgetc(infile);

      if (byte == EOF)
          break;

      fputc(byte, outfile);
  }

  // Process 32-bit samples
  size_t samples_to_read = chunk_size / sizeof(float);

  double x_prev1 = 0.0, y_prev1 = 0.0;
  double x_prev2 = 0.0, y_prev2 = 0.0;

  double x1_prev1 = 0.0, y1_prev1 = 0.0;
  double x1_prev2 = 0.0, y1_prev2 = 0.0;

  float sample_in;

  for (size_t i = 0; i < samples_to_read; i++) {

    if (fread(&sample_in, sizeof(float), 1, infile) != 1)
      break;

    double x = (double)sample_in;

    // First high-pass stage
    double y1 =
        b0 * x + b1 * x_prev1 + b2 * x_prev2 - a1 * y_prev1 - a2 * y_prev2;

    x_prev2 = x_prev1;
    x_prev1 = x;
    y_prev2 = y_prev1;
    y_prev1 = y1;
    // Second high-pass stage
    double y2;

    if (order == 4) {

        y2 = b0_1*y1 + b1_1*x1_prev1 + b2_1*x1_prev2 - a1_1*y1_prev1 - a2_1*y1_prev2;

        x1_prev2 = x1_prev1;
        x1_prev1 = y1;

        y1_prev2 = y1_prev1;
        y1_prev1 = y2;

    } else {

        y2 = y1;
    }

    float sample_out = (float)y2;

    fwrite(&sample_out, sizeof(float), 1, outfile);
  }

  fclose(infile);
  fclose(outfile);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void reverse_bilinear_lowpass(const char *input_filename,
                              const char *output_filename,
                              double cutoff_freq_hz, int order) {
  FILE *infile = fopen(input_filename, "rb");
  if (!infile) {
    perror("Failed to open input WAV");
    return;
  }

  WavHeader header;
  
  if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
    perror("Failed to read WAV header");
    fclose(infile);
    return;
}
  fseek(infile, 12, SEEK_SET);

  char chunk_id[4];
  uint32_t chunk_size;
  int found_data = 0;

  while (fread(chunk_id, 1, 4, infile) == 4) {

      if (fread(&chunk_size, sizeof(uint32_t), 1, infile) != 1)
        break;

      if (memcmp(chunk_id, "data", 4) == 0) {
        found_data = 1;
        printf("Found data chunk: %u bytes\n", chunk_size);

         break;
      }

      // Skip this chunk, including optional RIFF padding
      fseek(infile, chunk_size + (chunk_size & 1u), SEEK_CUR);
  }
  
  if (!found_data) {
    fprintf(stderr, "WAV data chunk not found\n");
    fclose(infile);
    return;
  } 
  
  // Save the audio starting position
  long audio_offset = ftell(infile);

  // Basic check  mono/stereo PCM
  double fs = (double)header.sampleRate;
  double T = 1.0 / fs;

  // Pre-warp cutoff frequency
  double omega_c = 2.0 * M_PI * cutoff_freq_hz;
  double Omega_c = (2.0 / T) * tan(omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double Q_multiplier = 1.0;

  
  double k = Omega_c * T / 2.0;

  double gamma, b0, b1, b2, a1, a2;
  double gamma1 = 0, b0_1 = 0, b1_1 = 0;
  double b2_1 = 0, a1_1 = 0, a2_1 = 0;

  if (order == 1) {

      gamma = 1.0 + k;

      b0 = k / gamma;
      b1 = k / gamma;
      b2 = 0.0;

      a1 = (k - 1.0) / gamma;
      a2 = 0.0;

  } else if (order == 2 || order == 4) {

      double Q = BUTTERWORTH_Q_LOOKUP[order][0] / Q_multiplier;

      gamma = 1.0 + k / Q + k*k;

      b0 = k*k / gamma;
      b1 = 2.0*k*k / gamma;
      b2 = k*k / gamma;

      a1 = 2.0*(k*k - 1.0) / gamma;
      a2 = (1.0 - k/Q + k*k) / gamma;

      if (order == 4) {

          double Q1 = BUTTERWORTH_Q_LOOKUP[4][1] / Q_multiplier;

          gamma1 = 1.0 + k/Q1 + k*k;

          b0_1 = k*k / gamma1;
          b1_1 = 2.0*k*k / gamma1;
          b2_1 = k*k / gamma1;

          a1_1 = 2.0*(k*k - 1.0) / gamma1;
          a2_1 = (1.0 - k/Q1 + k*k) / gamma1;
      }
  }
FILE *outfile = fopen(output_filename, "wb");
  if (!outfile) {
    fclose(infile);
    perror("Failed to open output WAV");
    return;
  }
  
  // Return to beginning of input WAV
  fseek(infile, 0, SEEK_SET);

  // Copy complete WAV header
  for (long i = 0; i < audio_offset; i++) {

      int byte = fgetc(infile);

      if (byte == EOF)
          break;

      fputc(byte, outfile);
  }

  // Process 32-bit samples
  size_t samples_to_read = chunk_size / sizeof(float);

  size_t N = 65536;

  size_t guard = 2048; // guard is n samples discarded at each end
  size_t hop = N - 2 * guard; // hop is the n of output samples gen per iteration

  double lambda = 0.001;
 
  // Allocate memory using fftw functions
  double *in = fftw_alloc_real(N);
  double *audio = fftw_alloc_real(samples_to_read);
  fftw_complex *out = fftw_alloc_complex(N / 2 + 1);

  // Create r2c plan
  fftw_plan forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

  fftw_plan inverse = fftw_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);
  
  // Read the entire WAV into memory
for (size_t i = 0; i < samples_to_read; i++) {

    float sample;

    if (fread(&sample, sizeof(float), 1, infile) == 1) {

        audio[i] = (double)sample;

    } else {

        audio[i] = 0.0;
    }
}
  
  for (size_t offset = 0; offset < samples_to_read; offset += hop) {
    
    for (size_t i = 0; i < N; i++) {

      int64_t sample_index = (int64_t)offset - (int64_t)guard + (int64_t)i;

      if (sample_index >= 0 && (uint64_t)sample_index < samples_to_read) {

        in[i] = audio[sample_index];

      } else {

        in[i] = 0.0;
      }
    }

    // Execute transform
    fftw_execute(forward);

    // apply inverse filt
    for (size_t bin = 0; bin < N / 2 + 1; bin++) {

      double omega = 2.0 * M_PI * bin / N;
      double complex ejw = cexp(-I * omega);
      double complex ej2w = cexp(-2.0 * I * omega);

      // First biquad
      double complex H1 =
          (b0 + b1 * ejw + b2 * ej2w) / (1.0 + a1 * ejw + a2 * ej2w);

      // Total fourth-order transfer function
      double complex H;

      if (order == 4) {
          // Second biquad
          double complex H2 = (b0_1 + b1_1*ejw + b2_1*ej2w) / (1.0 + a1_1*ejw + a2_1*ej2w);

          H = H1 * H2;

      } else {
          H = H1;
      }

      // Squared magnitude
      double magnitude_squared = creal(H * conj(H));

      // Regularized inverse
      double complex G = conj(H) / (magnitude_squared + lambda);
      // use this to get a 1:1 output, for debug and verif
      // double complex G = 1.0;
      out[bin] *= G;
    }

    // Execute transform
    fftw_execute(inverse);

    size_t valid = samples_to_read - offset;
    if (valid > hop)
      valid = hop;

    for (size_t i = 0; i < valid; i++) {

      double reconstructed = in[i + guard] / N;

      float sample_out = (float)reconstructed;

      fwrite(&sample_out, sizeof(float), 1, outfile);
    }
  }

  // Cleanup
  fftw_destroy_plan(forward);
  fftw_destroy_plan(inverse);

  fftw_free(in);
  fftw_free(out);
  fftw_free(audio);
  

  fclose(infile);
  fclose(outfile);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void reverse_bilinear_highpass(const char *input_filename,
                              const char *output_filename,
                              double cutoff_freq_hz, int order) {
  FILE *infile = fopen(input_filename, "rb");
  if (!infile) {
    perror("Failed to open input WAV");
    return;
  }

  WavHeader header;
  
  if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
    perror("Failed to read WAV header");
    fclose(infile);
    return;
}
  fseek(infile, 12, SEEK_SET);

  char chunk_id[4];
  uint32_t chunk_size;
  int found_data = 0;

  while (fread(chunk_id, 1, 4, infile) == 4) {

      if (fread(&chunk_size, sizeof(uint32_t), 1, infile) != 1)
        break;

      if (memcmp(chunk_id, "data", 4) == 0) {
        found_data = 1;
        printf("Found data chunk: %u bytes\n", chunk_size);

         break;
      }

      // Skip this chunk, including optional RIFF padding
      fseek(infile, chunk_size + (chunk_size & 1u), SEEK_CUR);
  }
  
  if (!found_data) {
    fprintf(stderr, "WAV data chunk not found\n");
    fclose(infile);
    return;
  } 
  
  // Save the audio starting position
  long audio_offset = ftell(infile);

  // Basic check  mono/stereo PCM
  double fs = (double)header.sampleRate;
  double T = 1.0 / fs;

  // Pre-warp cutoff frequency
  double omega_c = 2.0 * M_PI * cutoff_freq_hz;
  double Omega_c = (2.0 / T) * tan(omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double Q_multiplier = 1.0;

  
  double k = Omega_c * T / 2.0;

  double gamma, b0, b1, b2, a1, a2;
  double gamma1 = 0, b0_1 = 0, b1_1 = 0;
  double b2_1 = 0, a1_1 = 0, a2_1 = 0;
  if (order == 1) {

      	gamma = 1.0 + k;

	b0 = 1.0 / gamma;
	b1 = -1.0 / gamma;
	b2 = 0.0;

	a1 = (k - 1.0) / gamma;
	a2 = 0.0;

  } else if (order == 2 || order == 4) {

        gamma = 1.0 + (Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * k + k*k;

        b0 = 1 / gamma;
        b1 = -2 / gamma;
        b2 = 1 / gamma;

        a1 = 2 * (pow(k, 2) - 1.0) / gamma;
        a2 = (1.0 - (Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * k + k*k) / gamma;

      if (order == 4) {

          gamma1 = 1.0 + (Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1]) * k + pow(k, 2);

          b0_1 = 1 / gamma1;
          b1_1 = -2 / gamma1;
          b2_1 = 1 / gamma1;

          a1_1 = 2 * (pow(k, 2) - 1.0) / gamma1;
          a2_1 = (1.0 - Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1] * k + pow(k, 2)) /
      gamma1;
      }
  }

  FILE *outfile = fopen(output_filename, "wb");
  if (!outfile) {
    fclose(infile);
    perror("Failed to open output WAV");
    return;
  }
  
  // Return to beginning of input WAV
  fseek(infile, 0, SEEK_SET);

  // Copy complete WAV header
  for (long i = 0; i < audio_offset; i++) {

      int byte = fgetc(infile);

      if (byte == EOF)
          break;

      fputc(byte, outfile);
  }


  // Process 32-bit samples
  size_t samples_to_read = chunk_size / sizeof(float);

  size_t N = 65536;

  size_t guard = 2048; // guard is n samples discarded at each end
  size_t hop = N - 2 * guard; // hop is the n of output samples gen per iteration

  double lambda = 0.001;
 
  // Allocate memory using fftw functions
  double *in = fftw_alloc_real(N);
  double *audio = fftw_alloc_real(samples_to_read);
  fftw_complex *out = fftw_alloc_complex(N / 2 + 1);

  // Create r2c plan
  fftw_plan forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

  fftw_plan inverse = fftw_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);
  
  // Read the entire WAV into memory
  for (size_t i = 0; i < samples_to_read; i++) {

    float sample;

    if (fread(&sample, sizeof(float), 1, infile) == 1) {

        audio[i] = (double)sample;

    } else {

        audio[i] = 0.0;
    }
  } 
  
  for (size_t offset = 0; offset < samples_to_read; offset += hop) {

    for (size_t i = 0; i < N; i++) {

      int64_t sample_index = (int64_t)offset - (int64_t)guard + (int64_t)i;

      if (sample_index >= 0 && (uint64_t)sample_index < samples_to_read) {

        in[i] = audio[sample_index];

      } else {

        in[i] = 0.0;
      }
    }

    // Execute transform
    fftw_execute(forward);

    // apply inverse filt
    for (size_t bin = 0; bin < N / 2 + 1; bin++) {

      double omega = 2.0 * M_PI * bin / N;
      double complex ejw = cexp(-I * omega);
      double complex ej2w = cexp(-2.0 * I * omega);

      // First biquad
      double complex H1 =
          (b0 + b1 * ejw + b2 * ej2w) / (1.0 + a1 * ejw + a2 * ej2w);

      // Second biquad
     double complex H;

    if (order == 4) {

        double complex H2 = (b0_1 + b1_1*ejw + b2_1*ej2w) / (1.0 + a1_1*ejw + a2_1*ej2w);

        H = H1 * H2;

    } else {

        H = H1;
    }

      // Squared magnitude
      double magnitude_squared = creal(H * conj(H));

      // Regularized inverse
      double complex G = conj(H) / (magnitude_squared + lambda);
      // use this to get a 1:1 output, for debug and verif
      // double complex G = 1.0;
      out[bin] *= G;
    }

    // Execute transform
    fftw_execute(inverse);

    size_t valid = samples_to_read - offset;
    if (valid > hop)
      valid = hop;

    for (size_t i = 0; i < valid; i++) {

      double reconstructed = in[i + guard] / N;

      float sample_out = (float)reconstructed;

      fwrite(&sample_out, sizeof(float), 1, outfile);
    }
  }

  // Cleanup
  fftw_destroy_plan(forward);
  fftw_destroy_plan(inverse);

  fftw_free(in);
  fftw_free(out);
  fftw_free(audio);
  
  fclose(infile);
  fclose(outfile);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void reverse_bilinear_bandpass(const char *input_filename,
                              const char *output_filename,
                              double hp_cutoff_freq_hz, double lp_cutoff_freq_hz,  int order) {
  FILE *infile = fopen(input_filename, "rb");
  if (!infile) {
    perror("Failed to open input WAV");
    return;
  }

  WavHeader header;
  
  if (fread(&header, sizeof(WavHeader), 1, infile) != 1) {
    perror("Failed to read WAV header");
    fclose(infile);
    return;
}
  fseek(infile, 12, SEEK_SET);

  char chunk_id[4];
  uint32_t chunk_size;
  int found_data = 0;

  while (fread(chunk_id, 1, 4, infile) == 4) {

      if (fread(&chunk_size, sizeof(uint32_t), 1, infile) != 1)
        break;

      if (memcmp(chunk_id, "data", 4) == 0) {
        found_data = 1;
        printf("Found data chunk: %u bytes\n", chunk_size);

         break;
      }

      // Skip this chunk, including optional RIFF padding
      fseek(infile, chunk_size + (chunk_size & 1u), SEEK_CUR);
  }
  
  if (!found_data) {
    fprintf(stderr, "WAV data chunk not found\n");
    fclose(infile);
    return;
  } 
  
  // Save the audio starting position
  long audio_offset = ftell(infile);

  // Basic check  mono/stereo PCM
  double fs = (double)header.sampleRate;
  double T = 1.0 / fs;

  // HP Pre-warp cutoff frequency
  double hp_omega_c = 2.0 * M_PI * hp_cutoff_freq_hz;
  double hp_Omega_c = (2.0 / T) * tan(hp_omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double hp_Q_multiplier = 1.0;

  
  double hp_k = hp_Omega_c * T / 2.0;

  double hp_gamma, hp_b0, hp_b1, hp_b2, hp_a1, hp_a2;
  double hp_gamma1 = 0, hp_b0_1 = 0, hp_b1_1 = 0;
  double hp_b2_1 = 0, hp_a1_1 = 0, hp_a2_1 = 0;
  if (order == 1) {

      	hp_gamma = 1.0 + hp_k;

	hp_b0 = 1.0 / hp_gamma;
	hp_b1 = -1.0 / hp_gamma;
	hp_b2 = 0.0;

	hp_a1 = (hp_k - 1.0) / hp_gamma;
	hp_a2 = 0.0;

  } else if (order == 2 || order == 4) {

        hp_gamma = 1.0 + (hp_Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * hp_k + hp_k*hp_k;

        hp_b0 = 1 / hp_gamma;
        hp_b1 = -2 / hp_gamma;
        hp_b2 = 1 / hp_gamma;

        hp_a1 = 2 * (pow(hp_k, 2) - 1.0) / hp_gamma;
        hp_a2 = (1.0 - (hp_Q_multiplier / BUTTERWORTH_Q_LOOKUP[order][0]) * hp_k + hp_k*hp_k) / hp_gamma;

      if (order == 4) {

          hp_gamma1 = 1.0 + (hp_Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1]) * hp_k + pow(hp_k, 2);

          hp_b0_1 = 1 / hp_gamma1;
          hp_b1_1 = -2 / hp_gamma1;
          hp_b2_1 = 1 / hp_gamma1;

          hp_a1_1 = 2 * (pow(hp_k, 2) - 1.0) / hp_gamma1;
          hp_a2_1 = (1.0 - hp_Q_multiplier * 1 / BUTTERWORTH_Q_LOOKUP[4][1] * hp_k + pow(hp_k, 2)) /
      hp_gamma1;
      }
  }
      
  // Pre-warp cutoff frequency
  double lp_omega_c = 2.0 * M_PI * lp_cutoff_freq_hz;
  double lp_Omega_c = (2.0 / T) * tan(lp_omega_c * T / 2.0);

  // 1st order Butterworth bilinear coefficients for H_a(s) = Omega_c / (s +
  // Omega_c)
  double lp_Q_multiplier = 1.0;

  
  double lp_k = lp_Omega_c * T / 2.0;

  double lp_gamma, lp_b0, lp_b1, lp_b2, lp_a1, lp_a2;
  double lp_gamma1 = 0, lp_b0_1 = 0, lp_b1_1 = 0;
  double lp_b2_1 = 0, lp_a1_1 = 0, lp_a2_1 = 0;

  if (order == 1) {

      lp_gamma = 1.0 + lp_k;

      lp_b0 = lp_k / lp_gamma;
      lp_b1 = lp_k / lp_gamma;
      lp_b2 = 0.0;

      lp_a1 = (lp_k - 1.0) / lp_gamma;
      lp_a2 = 0.0;

  } else if (order == 2 || order == 4) {

      double lp_Q = BUTTERWORTH_Q_LOOKUP[order][0] / lp_Q_multiplier;

      lp_gamma = 1.0 + lp_k / lp_Q + lp_k*lp_k;

      lp_b0 = lp_k*lp_k / lp_gamma;
      lp_b1 = 2.0*lp_k*lp_k / lp_gamma;
      lp_b2 = lp_k*lp_k / lp_gamma;

      lp_a1 = 2.0*(lp_k*lp_k - 1.0) / lp_gamma;
      lp_a2 = (1.0 - lp_k/lp_Q + lp_k*lp_k) / lp_gamma;

      if (order == 4) {

          double lp_Q1 = BUTTERWORTH_Q_LOOKUP[4][1] / lp_Q_multiplier;

          lp_gamma1 = 1.0 + lp_k/lp_Q1 + lp_k*lp_k;

          lp_b0_1 = lp_k*lp_k / lp_gamma1;
          lp_b1_1 = 2.0*lp_k*lp_k / lp_gamma1;
          lp_b2_1 = lp_k*lp_k / lp_gamma1;

          lp_a1_1 = 2.0*(lp_k*lp_k - 1.0) / lp_gamma1;
          lp_a2_1 = (1.0 - lp_k/lp_Q1 + lp_k*lp_k) / lp_gamma1;
      }
  }

  FILE *outfile = fopen(output_filename, "wb");
  if (!outfile) {
    fclose(infile);
    perror("Failed to open output WAV");
    return;
  }
  
  // Return to beginning of input WAV
  fseek(infile, 0, SEEK_SET);

  // Copy complete WAV header
  for (long i = 0; i < audio_offset; i++) {

      int byte = fgetc(infile);

      if (byte == EOF)
          break;

      fputc(byte, outfile);
  }


  // Process 32-bit samples
  size_t samples_to_read = chunk_size / sizeof(float);

  size_t N = 65536;

  size_t guard = 2048; // guard is n samples discarded at each end
  size_t hop = N - 2 * guard; // hop is the n of output samples gen per iteration

  double lambda = 0.001;
 
  // Allocate memory using fftw functions
  double *in = fftw_alloc_real(N);
  double *audio = fftw_alloc_real(samples_to_read);
  fftw_complex *out = fftw_alloc_complex(N / 2 + 1);

  // Create r2c plan
  fftw_plan forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

  fftw_plan inverse = fftw_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);
  
  // Read the entire WAV into memory
  for (size_t i = 0; i < samples_to_read; i++) {

    float sample;

    if (fread(&sample, sizeof(float), 1, infile) == 1) {

        audio[i] = (double)sample;

    } else {

        audio[i] = 0.0;
    }
  } 
  
  for (size_t offset = 0; offset < samples_to_read; offset += hop) {

    for (size_t i = 0; i < N; i++) {

      int64_t sample_index = (int64_t)offset - (int64_t)guard + (int64_t)i;

      if (sample_index >= 0 && (uint64_t)sample_index < samples_to_read) {

        in[i] = audio[sample_index];

      } else {

        in[i] = 0.0;
      }
    }

    // Execute transform
    fftw_execute(forward);

    // apply inverse filt
    for (size_t bin = 0; bin < N / 2 + 1; bin++) {

      double omega = 2.0 * M_PI * bin / N;
      double complex ejw = cexp(-I * omega);
      double complex ej2w = cexp(-2.0 * I * omega);

      // First HP biquad
      double complex HP_H1 =
          (hp_b0 + hp_b1 * ejw + hp_b2 * ej2w) / (1.0 + hp_a1 * ejw + hp_a2 * ej2w);

      // Second HP biquad
      double complex HP_H2 =
          (hp_b0_1 + hp_b1_1 * ejw + hp_b2_1 * ej2w) / (1.0 + hp_a1_1 * ejw + hp_a2_1 * ej2w);
          
      // First LP biquad
      double complex LP_H1 =
          (lp_b0 + lp_b1 * ejw + lp_b2 * ej2w) / (1.0 + lp_a1 * ejw + lp_a2 * ej2w);

      // Second LP biquad
      double complex LP_H2 =
          (lp_b0_1 + lp_b1_1 * ejw + lp_b2_1 * ej2w) / (1.0 + lp_a1_1 * ejw + lp_a2_1 * ej2w);

      // Total eighth-order transfer function (two fourth order) 
      double complex H;

      if (order == 4) {

          H = HP_H1 * HP_H2 * LP_H1 * LP_H2;

      } else {

          H = HP_H1 * LP_H1;
      }

      // Squared magnitude
      double magnitude_squared = creal(H * conj(H));

      // Regularized inverse
      double complex G = conj(H) / (magnitude_squared + lambda);
      // use this to get a 1:1 output, for debug and verif
      // double complex G = 1.0;
      out[bin] *= G;
    }

    // Execute transform
    fftw_execute(inverse);

    size_t valid = samples_to_read - offset;
    if (valid > hop)
      valid = hop;

    for (size_t i = 0; i < valid; i++) {

      double reconstructed = in[i + guard] / N;

      float sample_out = (float)reconstructed;

      fwrite(&sample_out, sizeof(float), 1, outfile);
    }
  }

  // Cleanup
  fftw_destroy_plan(forward);
  fftw_destroy_plan(inverse);

  fftw_free(in);
  fftw_free(out);
  fftw_free(audio);
  
  fclose(infile);
  fclose(outfile);
}
