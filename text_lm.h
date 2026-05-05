#ifndef TEXT_LM_H
#define TEXT_LM_H

#include "transformer.h"
#include <stdio.h>

#define TEXT_LM_VOCAB_SIZE 28

typedef struct {
    int *token_ids;
    int len;
} TextCorpus;

typedef struct {
    int seq_len;
    int batch_size;
    int d_model;
    int d_ff;
    int nhead;
    int encoder_layers;
    int decoder_layers;
    int max_epochs;
    int early_stop_patience;
    float learning_rate;
    float dropout;
} TextLmHyperparams;

extern const TextLmHyperparams text_lm_default_hyperparams;

int text_corpus_from_text(const char *text, TextCorpus *out);
void text_corpus_free(TextCorpus *c);

int text_lm_train(const TextCorpus *corpus, const TextLmHyperparams *hp, Transformer **model_out,
                  TrainingState **ts_out);
void text_lm_free_session(Transformer *model, TrainingState *ts);

void text_lm_generate(Transformer *model, const TextLmHyperparams *hp, const char *seed, int length,
                      float temperature, FILE *out);

int text_lm_save(Transformer *model, TrainingState *ts, float lr, const char *path);
int text_lm_load(const char *path, Transformer **model_out, TrainingState **ts_out);

const char *text_lm_vocab_chars(void);
char text_lm_idx_to_char(int idx);
int text_lm_char_to_idx(char c);

#endif
