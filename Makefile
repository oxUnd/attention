CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC = transformer.c main.c text_lm.c train_copy_task.c train_simple.c train_full.c train_text.c
OBJ = $(SRC:.c=.o)
TARGET = transformer
TRAIN_TARGET = train_copy_task
SIMPLE_TARGET = train_simple
FULL_TARGET = train_full
TEXT_TARGET = train_text

all: $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(TEXT_TARGET)

$(TARGET): transformer.o main.o text_lm.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TRAIN_TARGET): transformer.o train_copy_task.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SIMPLE_TARGET): transformer.o train_simple.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(FULL_TARGET): transformer.o train_full.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TEXT_TARGET): transformer.o train_text.o text_lm.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.c text_lm.h transformer.h
text_lm.o: text_lm.c text_lm.h transformer.h
train_text.o: train_text.c text_lm.h transformer.h

train: $(FULL_TARGET)
	./$(FULL_TARGET)

test: $(TARGET)
	./$(TARGET) --help

clean:
	rm -f $(OBJ) $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(TEXT_TARGET)

.PHONY: all clean test train
