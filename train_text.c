#include "text_lm.h"
#include "tokenizer.h"
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Compact, deliberately repetitive corpus so the toy framework can actually
 * memorise short n-gram structure within a few minutes of CPU training.
 * Word vocabulary is small (~22 unique tokens). */
static const char CORPUS[] =
    "the cat sees the dog. the dog sees the cat. "
    "the cat is happy. the dog is happy. "
    "the cat is sad. the dog is sad. "
    "the cat sees a bird. the dog sees a bird. "
    "a bird sees the cat. a bird sees the dog. "
    "the cat is small. the dog is big. "
    "a small cat sees a big dog. a big dog sees a small cat. "
    "the bird is happy. the bird is sad. the bird is small. "
    "the cat is a friend. the dog is a friend. the bird is a friend. "
    "the friend sees the cat. the friend sees the dog. the friend sees the bird. "
    "the cat sees a friend. the dog sees a friend. the bird sees a friend. "
    "a friend is happy. a friend is sad. a friend is small. a friend is big. "
    "the cat sees the cat. the dog sees the dog. the bird sees the bird. "
    "the small bird is happy. the big bird is sad. "
    "the small dog sees the big cat. the big cat sees the small dog. ";

static int eval_held_out(Transformer *model, const TextLmHyperparams *hp, const TextCorpus *corpus,
                         int n_samples) {
    int seq_len = hp->seq_len;
    int d_model = hp->d_model;
    int vocab_size = hp->vocab_size;
    int n_starts = corpus->len - seq_len - 1;
    if (n_starts < 1) return -1;

    Tensor3D src = tensor_create(1, seq_len, d_model);
    Tensor3D tgt = tensor_create(1, seq_len, d_model);
    Tensor3D mask = causal_mask_create(seq_len);
    int *targets = (int *)malloc((size_t)seq_len * sizeof(int));

    float sum_loss = 0.0f;
    int correct = 0, total = 0;
    for (int i = 0; i < n_samples; i++) {
        int start = rand() % n_starts;
        tensor_zero(&src);
        tensor_zero(&tgt);
        for (int s = 0; s < seq_len; s++) {
            int ci = corpus->token_ids[start + s];
            if (ci >= 0 && ci < d_model) {
                src.data[s * d_model + ci] = 1.0f;
                tgt.data[s * d_model + ci] = 1.0f;
            }
            targets[s] = corpus->token_ids[start + s + 1];
        }
        Tensor3D out = transformer_forward(model, &src, &tgt, &mask, &mask, NULL);
        LossResult lr = cross_entropy_loss(&out, targets, vocab_size);
        sum_loss += lr.loss;

        for (int s = 0; s < seq_len; s++) {
            int offset = s * d_model;
            int pred = 0;
            float mv = out.data[offset];
            for (int v = 1; v < vocab_size && v < d_model; v++) {
                if (out.data[offset + v] > mv) {
                    mv = out.data[offset + v];
                    pred = v;
                }
            }
            if (pred == targets[s]) correct++;
            total++;
        }
        tensor_free(&out);
        tensor_free(&lr.grad);
    }
    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&mask);
    free(targets);
    printf("Held-out eval (%d windows): loss=%.4f acc=%.1f%%\n", n_samples,
           sum_loss / (float)n_samples, 100.0f * (float)correct / (float)total);
    return 0;
}

static void run_word_demo(void) {
    printf("\n========================================\n");
    printf(" Demo 1: word-level LM (built-in corpus)\n");
    printf("========================================\n\n");

    Tokenizer *tok = tokenizer_create_word(CORPUS, 64, 1);
    tokenizer_print_stats(tok, stdout);

    TextCorpus corpus = {0};
    if (text_corpus_from_text_with_tokenizer(CORPUS, tok, &corpus) != 0) {
        fprintf(stderr, "Word-level corpus encoding failed.\n");
        tokenizer_free(tok);
        return;
    }
    printf("Corpus tokens: %d\n", corpus.len);

    TextLmHyperparams hp = text_lm_default_hyperparams;
    hp.d_model = 64;
    hp.d_ff = 128;
    hp.nhead = 4;
    hp.encoder_layers = 1;
    hp.decoder_layers = 1;
    hp.seq_len = 8;
    hp.batch_size = 16;
    hp.learning_rate = 0.003f;
    hp.max_epochs = 60;
    hp.early_stop_patience = 12;
    hp.vocab_size = tok->vocab_size;
    if (hp.vocab_size > hp.d_model) hp.vocab_size = hp.d_model;

    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_train(&corpus, &hp, &model, &ts) != 0) {
        text_corpus_free(&corpus);
        tokenizer_free(tok);
        return;
    }

    eval_held_out(model, &hp, &corpus, 32);

    printf("\n--- Word-level samples (T=0.4) ---\n");
    text_lm_generate_with_tokenizer(model, &hp, tok, "the cat", 16, 0.4f, stdout);
    text_lm_generate_with_tokenizer(model, &hp, tok, "the dog", 16, 0.4f, stdout);
    text_lm_generate_with_tokenizer(model, &hp, tok, "a small", 16, 0.4f, stdout);
    printf("--- Word-level samples (T=0.8) ---\n");
    text_lm_generate_with_tokenizer(model, &hp, tok, "the cat", 16, 0.8f, stdout);

    text_corpus_free(&corpus);
    text_lm_free_session(model, ts);
    tokenizer_free(tok);
}

static void run_char_demo(void) {
    printf("\n========================================\n");
    printf(" Demo 2: char-level LM (a-z, ., space)\n");
    printf("========================================\n\n");

    TextLmHyperparams hp = text_lm_default_hyperparams;
    hp.d_model = 64;
    hp.d_ff = 128;
    hp.nhead = 4;
    hp.encoder_layers = 1;
    hp.decoder_layers = 1;
    hp.seq_len = 16;
    hp.batch_size = 16;
    hp.learning_rate = 0.003f;
    hp.max_epochs = 30;
    hp.early_stop_patience = 8;
    hp.vocab_size = TEXT_LM_VOCAB_SIZE;

    TextCorpus corpus = {0};
    if (text_corpus_from_text(CORPUS, &corpus) != 0) {
        fprintf(stderr, "Corpus preprocess failed.\n");
        return;
    }
    printf("Corpus chars: %d, vocab: %d (%s)\n", corpus.len, TEXT_LM_VOCAB_SIZE, text_lm_vocab_chars());

    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_train(&corpus, &hp, &model, &ts) != 0) {
        text_corpus_free(&corpus);
        return;
    }

    eval_held_out(model, &hp, &corpus, 32);

    printf("\n--- Char-level samples (T=0.4) ---\n");
    text_lm_generate(model, &hp, "the cat ", 80, 0.4f, stdout);
    printf("--- Char-level samples (T=0.8) ---\n");
    text_lm_generate(model, &hp, "the dog ", 80, 0.8f, stdout);

    text_corpus_free(&corpus);
    text_lm_free_session(model, ts);
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    int run_word = 1;
    int run_char = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--word-only") == 0) {
            run_char = 0;
        } else if (strcmp(argv[i], "--char-only") == 0) {
            run_word = 0;
        }
    }

    printf("=== Transformer Text Training (built-in corpus) ===\n");
    printf("Demonstrates causal-mask LM training, char + word tokenizers,\n");
    printf("shifted next-token prediction, and held-out evaluation.\n");

    if (run_word) run_word_demo();
    if (run_char) run_char_demo();
    return 0;
}
