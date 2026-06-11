.PHONY: all clean

all:
	gcc -O2 -mavx2 -mfma ge.c -o ge

clean:
	rm -f ge