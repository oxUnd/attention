CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC = transformer.c main.c train_copy_task.c train_simple.c train_full.c
OBJ = $(SRC:.c=.o)
TARGET = transformer
TRAIN_TARGET = train_copy_task
SIMPLE_TARGET = train_simple
FULL_TARGET = train_full

all: $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET)

$(TARGET): transformer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TRAIN_TARGET): transformer.o train_copy_task.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SIMPLE_TARGET): transformer.o train_simple.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(FULL_TARGET): transformer.o train_full.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

train: $(FULL_TARGET)
	./$(FULL_TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET)

.PHONY: all clean test train
