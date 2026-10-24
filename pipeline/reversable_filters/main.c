#include "Butterworth.h" 
#include <stdio.h>
#include <stdint.h>





int main(void) {
        
        double fc_hp = 0;
        double fc_lp = 0;
        double AI_fc_hp = 0;
        double AI_fc_lp = 0;
        char inFile[100];
        char outFile[100];
        
        while(1){
          int choice = 0;
          
          
          printf("1 selects low pass, 2 selects high pass, 3 selects both\r\n");
          printf("4 selects low pass reverseal, 5 selects high pass reversal, 6 selects both reversal\r\n\r\n");
          printf("AI agent shall use options 7 through 9 to attempt to guess the reverse filter poles\r\n");
       
          printf("7 selects low pass reverseal, 8 selects high pass reversal, 9 selects both reversal\r\n\r\n");
          
          scanf("%d", &choice);
          if (choice == 1){
            printf("Enter low pass cutoff frequency (1Hz to 20kHz): ");
            scanf("%lf", &fc_lp);
            printf("\r\n");
            apply_bilinear_lowpass("voice_capture.wav","voice_capture_lp.wav", fc_lp);
          }
          else if (choice == 2){
            printf("Enter high pass cutoff frequency (1Hz to 20kHz): ");
            scanf("%lf", &fc_hp);
            printf("\r\n");
            apply_bilinear_highpass("voice_capture.wav","voice_capture_hp.wav", fc_hp);
          }
          else if (choice == 3){
            printf("Enter high pass cutoff frequency (1Hz to 20kHz): ");
            scanf("%lf", &fc_hp);
            printf("\r\n");
            apply_bilinear_highpass("voice_capture.wav","voice_capture_int.wav", fc_hp);
            
            printf("Enter low pass cutoff frequency (1Hz to 20kHz): ");
            scanf("%lf", &fc_lp);
            printf("\r\n");
            apply_bilinear_lowpass("voice_capture_int.wav","voice_capture_both.wav", fc_lp);
          }
          else if (choice == 4){
            printf("Attempting reversal of last Lowpass process...\r\n");
            reverse_bilinear_lowpass("voice_capture_lp.wav","voice_capture_reversed_lp.wav", fc_lp);
          }
          else if (choice == 5){
            printf("Attempting reversal of last highpass process...\r\n");
            reverse_bilinear_highpass("voice_capture_hp.wav","voice_capture_reversed_hp.wav", fc_hp);
          }
          else if (choice == 6){
            printf("Attempting reversal of both high and low pass filters\r\n");
            reverse_bilinear_lowpass("voice_capture_both.wav","voice_capture_reversed_int.wav", fc_lp);
            reverse_bilinear_highpass("voice_capture_reversed_int.wav","voice_capture_reversed_both.wav", fc_hp);
          }
          else if (choice == 7){
          
              printf("Enter inFile: ");
              if (scanf("%99s", inFile) == 1) {
                FILE *file = fopen(inFile, "r");
                if (file != NULL) {
                printf("File opened successfully.\n");
                fclose(file);
              } else {
                printf("Error opening file.\n");
                }
            }
            printf("Enter outFile: ");
            scanf("%99s", outFile);
            printf("\r\nEnter guess for Low pass cutoff frequency in Hz: ");
            scanf("%lf", &AI_fc_lp);
            printf("\r\n");
            reverse_bilinear_lowpass(inFile,outFile, AI_fc_lp);
          }
          else if (choice == 8){
          
              printf("Enter inFile: ");
              if (scanf("%99s", inFile) == 1) {
                FILE *file = fopen(inFile, "r");
                if (file != NULL) {
                printf("File opened successfully.\n");
                fclose(file);
              } else {
                printf("Error opening file.\n");
                }
            }
            printf("Enter outFile: ");
            scanf("%99s", outFile);
            printf("\r\nEnter guess for High pass cutoff frequency in Hz: ");
            scanf("%lf", &AI_fc_hp);
            printf("\r\n");
            reverse_bilinear_highpass(inFile,outFile, AI_fc_hp);
          }
          else if (choice == 9){
                      
              printf("Enter inFile: ");
              if (scanf("%99s", inFile) == 1) {
                FILE *file = fopen(inFile, "r");
                if (file != NULL) {
                printf("File opened successfully.\n");
                fclose(file);
              } else {
                printf("Error opening file.\n");
                }
            }
            printf("Enter outFile: ");
            scanf("%99s", outFile);                
            printf("\r\nEnter guess for Low pass cutoff frequency in Hz: ");
            scanf("%lf", &AI_fc_lp);
            printf("\r\n");
            reverse_bilinear_lowpass(inFile,"AI_int.wav", AI_fc_lp);   
            printf("Enter guess for High pass cutoff frequency in Hz: ");
            scanf("%lf", &AI_fc_hp);
            printf("\r\n");
            reverse_bilinear_highpass("AI_int.wav",outFile, AI_fc_hp);
          }
           
         }
        
    };
