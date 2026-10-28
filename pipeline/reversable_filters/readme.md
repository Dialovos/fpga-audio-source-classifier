CLI syntax:
./filter <filter type> <filter family> <filter order(1,2,4 are the only ones supported)> <inFile> <outFile> <menu option>  
CLI test commands:
./filter LP Butterworth 4 300 5000 voice_capture_both.wav voice_capture_br.wav 6
./filter LP Butterworth 4 300 5000 voice_capture.wav voice_capture_both.wav 3


Menu option 1 selects low pass, 2 selects high pass, 3 selects both (band pass)
4 selects low pass reverseal, 5 selects high pass reversal, 6 selects bandpass reversal '

Currently filter family and type are unused, but able to be passed into the program from external calls. 

Depending on how the program is called, the intermediate file (int.wav) could possibly get overwritten if multiple instances 
are run in parallel. report this if this is an issue. 
