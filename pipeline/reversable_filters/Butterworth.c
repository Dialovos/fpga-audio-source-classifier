#include "Butterworth.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int choice = 0;
  char *AI_type;
  char *AI_family;

  int AI_order;

  double AI_fc_hp;
  double AI_fc_lp;

  char *AI_inFile;
  char *AI_outFile;

  int AI_option;

  int mode = 0;

  if (argc == 9) {
    AI_type = argv[1];
    AI_family = argv[2];

    AI_order = atoi(argv[3]);

    AI_fc_hp = atof(argv[4]);
    AI_fc_lp = atof(argv[5]);

    AI_inFile = argv[6];
    AI_outFile = argv[7];

    choice = atoi(argv[8]);
    printf("CLI mode selected\r\n");
    mode = 1;
  } else {
    printf("Terminal mode selected (manual)\r\n");
  }

  double fc_hp = 0;
  double fc_lp = 0;
  char inFile[100];
  char outFile[100];
  int order = 4;

  while (1) {
    if (!mode) {
      printf("1 selects low pass, 2 selects high pass, 3 selects both\r\n");
      printf("4 selects low pass reverseal, 5 selects high pass reversal, 6 "
             "selects both reversal\r\n\r\n");
      printf("AI agent shall use options 7 through 9 to attempt to guess the "
             "reverse filter poles\r\n");

      printf("7 selects low pass reverseal, 8 selects high pass reversal, 9 "
             "selects both reversal\r\n\r\n");

      scanf("%d", &choice);
      if (choice == 1) {
        printf("Enter low pass cutoff frequency (1Hz to 20kHz): ");
        scanf("%lf", &fc_lp);
        printf("\r\n");
        apply_bilinear_lowpass("voice_capture.wav", "voice_capture_lp.wav",
                               fc_lp, order);

      } else if (choice == 2) {
        printf("Enter high pass cutoff frequency (1Hz to 20kHz): ");
        scanf("%lf", &fc_hp);
        printf("\r\n");
        apply_bilinear_highpass("voice_capture.wav", "voice_capture_hp.wav",
                                fc_hp, order);
      } else if (choice == 3) {
        printf("Enter high pass cutoff frequency (1Hz to 20kHz): ");
        scanf("%lf", &fc_hp);
        printf("\r\n");
        apply_bilinear_highpass("voice_capture.wav", "voice_capture_int.wav",
                                fc_hp, order);

        printf("Enter low pass cutoff frequency (1Hz to 20kHz): ");
        scanf("%lf", &fc_lp);
        printf("\r\n");
        apply_bilinear_lowpass("voice_capture_int.wav",
                               "voice_capture_both.wav", fc_lp, order);
      } else if (choice == 4) {
        printf("Attempting reversal of last Lowpass process...\r\n");
        reverse_bilinear_lowpass("voice_capture_lp.wav",
                                 "voice_capture_reversed_lp.wav", fc_lp, order);
      } else if (choice == 5) {
        printf("Attempting reversal of last highpass process...\r\n");
        reverse_bilinear_highpass("voice_capture_hp.wav",
                                  "voice_capture_reversed_hp.wav", fc_hp, order);
      } else if (choice == 6) {
        printf("Attempting reversal of both high and low pass filters\r\n");
        reverse_bilinear_lowpass("voice_capture_both.wav",
                                 "voice_capture_reversed_int.wav", fc_lp, order);
        reverse_bilinear_highpass("voice_capture_reversed_int.wav",
                                  "voice_capture_reversed_both.wav", fc_hp, order);
      }

      // menu for automated runs
    } else if (mode == 1 && strcmp(AI_family, "butter") == 0) {
      if (choice == 1) {
        apply_bilinear_lowpass(AI_inFile, AI_outFile, AI_fc_lp, AI_order);
        printf("applied lowpass\r\n");
      } else if (choice == 2) {
        apply_bilinear_highpass(AI_inFile, AI_outFile, AI_fc_lp, AI_order);
        printf("applied highpass\r\n");
      } else if (choice == 3) {
        apply_bilinear_highpass(AI_inFile, "int.wav", AI_fc_hp, AI_order);
        apply_bilinear_lowpass("int.wav", AI_outFile, AI_fc_lp, AI_order);
        printf("applied bandpass\r\n");
      } else if (choice == 4) {
        reverse_bilinear_lowpass(AI_inFile, AI_outFile, AI_fc_lp, AI_order);
        printf("reversed lowpass\r\n");
      } else if (choice == 5) {
        reverse_bilinear_highpass(AI_inFile, AI_outFile, AI_fc_hp, AI_order);
        printf("reversed highpass\r\n");
      } else if (choice == 6) {
        reverse_bilinear_bandpass(AI_inFile, AI_outFile, AI_fc_hp, AI_fc_lp, AI_order);
        printf("reversed bandpass\r\n");
        
      }
      printf("Automated task finished\r\n");
      return 1;
    } else if (mode == 1 && strcmp(AI_family, "butter") != 0){
      printf("invalid filter family\r\n");
      return -1;
    }
  }
  return 0;
};
