CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
SRC = nn_math.c transformer.c main.c text_lm.c tokenizer.c train_copy_task.c train_simple.c train_full.c train_text.c
OBJ = $(SRC:.c=.o)
TARGET = transformer
TRAIN_TARGET = train_copy_task
SIMPLE_TARGET = train_simple
FULL_TARGET = train_full
TEXT_TARGET = train_text

all: $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(TEXT_TARGET)

$(TARGET): nn_math.o transformer.o main.o text_lm.o tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TRAIN_TARGET): nn_math.o transformer.o train_copy_task.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SIMPLE_TARGET): nn_math.o transformer.o train_simple.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(FULL_TARGET): nn_math.o transformer.o train_full.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TEXT_TARGET): nn_math.o transformer.o train_text.o text_lm.o tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c nn_math.h transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

nn_math.o: nn_math.c nn_math.h
transformer.o: transformer.c transformer.h nn_math.h
main.o: main.c text_lm.h tokenizer.h transformer.h nn_math.h
text_lm.o: text_lm.c text_lm.h tokenizer.h transformer.h nn_math.h
tokenizer.o: tokenizer.c tokenizer.h
train_text.o: train_text.c text_lm.h tokenizer.h transformer.h nn_math.h

train: $(FULL_TARGET)
	./$(FULL_TARGET)

text: $(TEXT_TARGET)
	./$(TEXT_TARGET)

test: $(TARGET)
	./$(TARGET) --help

clean:
	rm -f $(OBJ) $(TARGET) $(TRAIN_TARGET) $(SIMPLE_TARGET) $(FULL_TARGET) $(TEXT_TARGET)

.PHONY: all clean test train text
