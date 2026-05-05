CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC = transformer.c main.c train_copy_task.c train_simple.c train_full.c train_text.c test_gradient.c
OBJ = $(SRC:.c=.o)
TARGET = transformer
TRAIN_TARGET = train_copy_task
SIMPLE_TARGET = train_simple
FULL_TARGET = train_full
TEXT_TARGET = train_text
GRADIENT_TARGET = test_gradient

all: $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(TEXT_TARGET) $(GRADIENT_TARGET)

$(TARGET): transformer.o main.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TRAIN_TARGET): transformer.o train_copy_task.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SIMPLE_TARGET): transformer.o train_simple.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(FULL_TARGET): transformer.o train_full.o
	$(CC) $(CFLAGS) -o $@ $^ -lm


$(TEXT_TARGET): transformer.o train_text.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(GRADIENT_TARGET): transformer.o test_gradient.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

train: $(FULL_TARGET)
	./$(FULL_TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(GRADIENT_TARGET)

.PHONY: all clean test train
