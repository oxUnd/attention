CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC = transformer.c main.c
OBJ = $(SRC:.c=.o)
TARGET = transformer

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean test
