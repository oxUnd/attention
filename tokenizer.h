#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>

typedef enum {
    TOKENIZER_CHAR = 0,
    TOKENIZER_BYTE = 1,
    TOKENIZER_WORD = 2,
} TokenizerKind;

#define TOK_PAD 0
#define TOK_UNK 1
#define TOK_BOS 2
#define TOK_EOS 3
#define TOK_NUM_SPECIALS 4

typedef struct TokenizerEntry {
    char *token;
    int id;
    struct TokenizerEntry *next;
} TokenizerEntry;

typedef struct {
    TokenizerKind kind;
    int vocab_size;
    int max_token_len;
    char **id_to_token;
    int char_to_id[256];
    TokenizerEntry **buckets;
    int n_buckets;
    int lowercase;
} Tokenizer;

Tokenizer *tokenizer_create_char(const char *charset, int lowercase);
Tokenizer *tokenizer_create_byte(void);
Tokenizer *tokenizer_create_word(const char *corpus, int max_vocab, int lowercase);
void tokenizer_free(Tokenizer *t);

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out);
int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int max_out);
const char *tokenizer_id_to_token(const Tokenizer *t, int id);
int tokenizer_token_to_id(const Tokenizer *t, const char *token);

int tokenizer_save(const Tokenizer *t, const char *path);
Tokenizer *tokenizer_load(const char *path);

void tokenizer_print_stats(const Tokenizer *t, FILE *out);

#endif
