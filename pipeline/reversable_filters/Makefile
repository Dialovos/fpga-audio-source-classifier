CC = gcc

CFLAGS = -Wall -Wextra -O2 -std=c11

TARGET = Butterworth_Process

OBJS = main.o Butterworth.o 


$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lfftw3 -lm


main.o: main.c Butterworth.h 
	$(CC) $(CFLAGS) -c main.c


LP_Butterworth.o: Butterworth.c Butterworth.h
	$(CC) $(CFLAGS) -c Butterworth.c


clean:
	rm -f $(OBJS) $(TARGET) 
