#ifndef TEXT_LM_H
#define TEXT_LM_H

#include "tokenizer.h"
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
    int num_heads;
    int encoder_layers;
    int decoder_layers;
    int max_epochs;
    int early_stop_patience;
    float learning_rate;
    float dropout;
    int vocab_size;
    /* Optional: when non-NULL, the trainer dumps the model+ts to this path
     * every time avg_loss reaches a new best. After training finishes, the
     * file therefore reflects the best-loss checkpoint (not the final epoch).
     * Leave NULL to disable auto-saving (legacy behaviour). */
    const char *best_checkpoint_path;
    /* Linear warmup for the first `warmup_epochs` epochs, then cosine decay
     * from learning_rate down to learning_rate * min_lr_factor over the
     * remaining epochs. When both are set to their defaults
     * (warmup_epochs == 0 && min_lr_factor == 1.0) the schedule is disabled
     * and the optimiser uses a constant learning rate. */
    int warmup_epochs;
    float min_lr_factor;
} TextLmHyperparams;

extern const TextLmHyperparams text_lm_default_hyperparams;

int text_corpus_from_text(const char *text, TextCorpus *out);
int text_corpus_from_text_with_tokenizer(const char *text, const Tokenizer *tok, TextCorpus *out);
void text_corpus_free(TextCorpus *c);

int text_lm_train(const TextCorpus *corpus, const TextLmHyperparams *hp, Transformer **model_out,
                  TrainingState **ts_out);
void text_lm_free_session(Transformer *model, TrainingState *ts);

void text_lm_generate(Transformer *model, const TextLmHyperparams *hp, const char *seed, int length,
                      float temperature, FILE *out);

void text_lm_generate_with_tokenizer(Transformer *model, const TextLmHyperparams *hp,
                                     const Tokenizer *tok, const char *seed, int length,
                                     float temperature, FILE *out);

/* KV-cache-accelerated variant. Encoder runs once (over the encoded seed)
 * and cross-attn K/V are pre-projected; each generated token only triggers
 * a single decoder step. The encoder context is therefore frozen at the
 * seed (it does NOT roll forward as new tokens are produced), which is a
 * minor distribution mismatch vs. the non-cached path but makes the cost
 * O(n) instead of O(n^2) per token in the decoder.
 *
 * Falls back to text_lm_generate_with_tokenizer when the model was not
 * built with vocab_size > 0 (legacy d_model-space path). */
void text_lm_generate_with_tokenizer_cached(Transformer *model,
                                            const TextLmHyperparams *hp,
                                            const Tokenizer *tok,
                                            const char *seed, int length,
                                            float temperature, FILE *out);

int text_lm_save(Transformer *model, TrainingState *ts, float lr, const char *path);
int text_lm_load(const char *path, Transformer **model_out, TrainingState **ts_out);

const char *text_lm_vocab_chars(void);
char text_lm_idx_to_char(int idx);
int text_lm_char_to_idx(char c);

#endif
