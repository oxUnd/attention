#include "text_lm.h"
#include "tokenizer.h"
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *read_text_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    if (out_len) *out_len = n;
    return buf;
}

static void hyperparams_from_model(const Transformer *m, TextLmHyperparams *hp) {
    *hp = text_lm_default_hyperparams;
    hp->d_model = m->config.d_model;
    hp->d_ff = m->config.d_ff;
    hp->num_heads = m->config.num_heads;
    hp->encoder_layers = m->config.encoder_layers;
    hp->decoder_layers = m->config.decoder_layers;
    hp->dropout = (float)m->config.dropout;
    if (m->config.max_len > 5) hp->seq_len = m->config.max_len - 5;
}

static void make_tok_path(const char *model_path, char *out, size_t cap) {
    snprintf(out, cap, "%s.tok", model_path);
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --train <corpus.txt> [options]\n"
            "  %s --load <model.bin>  [--generate <seed>] [options]\n"
            "\n"
            "Training options:\n"
            "  --tokenizer <char|word|byte>  Tokenizer (default: char).\n"
            "  --vocab-size <n>              Cap vocab when --tokenizer word (default 60).\n"
            "  --d-model <n>                 Model width (default 64; must be >= vocab).\n"
            "  --layers <enc>,<dec>          Encoder/decoder layer counts.\n"
            "  --seq-len <n>                 Window length (default 24).\n"
            "  --batch <n>                   Batch size (default 8).\n"
            "  --epochs <n>                  Max epochs (default 80).\n"
            "  --lr <f>                      Learning rate (default 0.003).\n"
            "  --out <path>                  Save model checkpoint (and <path>.tok).\n"
            "\n"
            "Generation options (used after --train or --load):\n"
            "  --generate <seed>             Sample continuation.\n"
            "  --length <n>                  Steps to generate (default 60).\n"
            "  --temperature <t>             Sampling temperature (default 0.8).\n",
            argv0, argv0);
}

typedef struct {
    const char *train_path;
    const char *out_path;
    const char *load_path;
    const char *generate_seed;
    const char *tok_kind;
    int vocab_cap;
    int d_model;
    int seq_len;
    int batch;
    int epochs;
    float lr;
    int enc_layers;
    int dec_layers;
    int gen_length;
    float temperature;
} CliArgs;

static int parse_args(int argc, char **argv, CliArgs *a) {
    a->train_path = NULL;
    a->out_path = NULL;
    a->load_path = NULL;
    a->generate_seed = NULL;
    a->tok_kind = "char";
    a->vocab_cap = 60;
    a->d_model = 0;
    a->seq_len = 0;
    a->batch = 0;
    a->epochs = 0;
    a->lr = 0.0f;
    a->enc_layers = 0;
    a->dec_layers = 0;
    a->gen_length = 60;
    a->temperature = 0.8f;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (!strcmp(k, "--help") || !strcmp(k, "-h")) return 1;
        if (!strcmp(k, "--train") && i + 1 < argc) { a->train_path = argv[++i]; continue; }
        if (!strcmp(k, "--out") && i + 1 < argc) { a->out_path = argv[++i]; continue; }
        if (!strcmp(k, "--load") && i + 1 < argc) { a->load_path = argv[++i]; continue; }
        if (!strcmp(k, "--generate") && i + 1 < argc) { a->generate_seed = argv[++i]; continue; }
        if (!strcmp(k, "--tokenizer") && i + 1 < argc) { a->tok_kind = argv[++i]; continue; }
        if (!strcmp(k, "--vocab-size") && i + 1 < argc) { a->vocab_cap = atoi(argv[++i]); continue; }
        if (!strcmp(k, "--d-model") && i + 1 < argc) { a->d_model = atoi(argv[++i]); continue; }
        if (!strcmp(k, "--seq-len") && i + 1 < argc) { a->seq_len = atoi(argv[++i]); continue; }
        if (!strcmp(k, "--batch") && i + 1 < argc) { a->batch = atoi(argv[++i]); continue; }
        if (!strcmp(k, "--epochs") && i + 1 < argc) { a->epochs = atoi(argv[++i]); continue; }
        if (!strcmp(k, "--lr") && i + 1 < argc) { a->lr = (float)strtod(argv[++i], NULL); continue; }
        if (!strcmp(k, "--length") && i + 1 < argc) {
            a->gen_length = atoi(argv[++i]);
            if (a->gen_length < 1) a->gen_length = 60;
            continue;
        }
        if (!strcmp(k, "--temperature") && i + 1 < argc) {
            a->temperature = (float)strtod(argv[++i], NULL);
            continue;
        }
        if (!strcmp(k, "--layers") && i + 1 < argc) {
            const char *v = argv[++i];
            const char *comma = strchr(v, ',');
            if (!comma) {
                fprintf(stderr, "--layers expects <enc>,<dec>\n");
                return -1;
            }
            a->enc_layers = atoi(v);
            a->dec_layers = atoi(comma + 1);
            continue;
        }
        fprintf(stderr, "Unknown argument: %s\n", k);
        return -1;
    }
    return 0;
}

static Tokenizer *make_tokenizer(const char *kind, const char *corpus, int vocab_cap) {
    if (!kind || !strcmp(kind, "char")) {
        return tokenizer_create_char(" .,'!?abcdefghijklmnopqrstuvwxyz", 1);
    }
    if (!strcmp(kind, "byte")) {
        return tokenizer_create_byte();
    }
    if (!strcmp(kind, "word")) {
        return tokenizer_create_word(corpus, vocab_cap, 1);
    }
    fprintf(stderr, "Unknown tokenizer: %s (use char|word|byte)\n", kind);
    return NULL;
}

int main(int argc, char **argv) {
    CliArgs a;
    int pa = parse_args(argc, argv, &a);
    if (pa == 1) { print_usage(argv[0]); return 0; }
    if (pa < 0) { print_usage(argv[0]); return 1; }

    if (!a.train_path && !a.load_path) { print_usage(argv[0]); return 1; }
    if (a.train_path && a.load_path) {
        fprintf(stderr, "Use either --train or --load, not both.\n");
        return 1;
    }

    srand((unsigned)time(NULL));

    if (a.train_path) {
        size_t raw_len = 0;
        char *text = read_text_file(a.train_path, &raw_len);
        if (!text) {
            fprintf(stderr, "Cannot read training file: %s\n", a.train_path);
            return 1;
        }

        Tokenizer *tok = make_tokenizer(a.tok_kind, text, a.vocab_cap);
        if (!tok) { free(text); return 1; }
        printf("=== Train from %s ===\n", a.train_path);
        tokenizer_print_stats(tok, stdout);

        TextCorpus corpus = {0};
        if (text_corpus_from_text_with_tokenizer(text, tok, &corpus) != 0) {
            fprintf(stderr, "Failed to encode corpus.\n");
            tokenizer_free(tok);
            free(text);
            return 1;
        }

        TextLmHyperparams hp = text_lm_default_hyperparams;
        if (a.d_model > 0) hp.d_model = a.d_model;
        if (a.seq_len > 0) hp.seq_len = a.seq_len;
        if (a.batch > 0) hp.batch_size = a.batch;
        if (a.epochs > 0) hp.max_epochs = a.epochs;
        if (a.lr > 0) hp.learning_rate = a.lr;
        if (a.enc_layers > 0) hp.encoder_layers = a.enc_layers;
        if (a.dec_layers > 0) hp.decoder_layers = a.dec_layers;
        hp.vocab_size = tok->vocab_size;

        if (hp.vocab_size > hp.d_model) {
            fprintf(stderr,
                    "vocab_size (%d) > d_model (%d). Increase --d-model or shrink --vocab-size.\n",
                    hp.vocab_size, hp.d_model);
            text_corpus_free(&corpus);
            tokenizer_free(tok);
            free(text);
            return 1;
        }

        printf("Hyperparams: seq=%d batch=%d d_model=%d d_ff=%d heads=%d enc=%d dec=%d epochs=%d lr=%.4f\n",
               hp.seq_len, hp.batch_size, hp.d_model, hp.d_ff, hp.num_heads, hp.encoder_layers,
               hp.decoder_layers, hp.max_epochs, hp.learning_rate);
        printf("Corpus tokens: %d (from %zu raw bytes)\n\n", corpus.len, raw_len);

        Transformer *model = NULL;
        TrainingState *ts = NULL;
        int tr = text_lm_train(&corpus, &hp, &model, &ts);
        text_corpus_free(&corpus);
        free(text);

        if (tr != 0) {
            fprintf(stderr, "Training failed (%d). Corpus may be too small for seq_len/batch.\n", tr);
            if (model) text_lm_free_session(model, ts);
            tokenizer_free(tok);
            return 1;
        }
        printf("\n=== Training complete ===\n");

        if (a.out_path) {
            if (text_lm_save(model, ts, hp.learning_rate, a.out_path) != 0) {
                fprintf(stderr, "Failed to write model file: %s\n", a.out_path);
                text_lm_free_session(model, ts);
                tokenizer_free(tok);
                return 1;
            }
            char tok_path[1024];
            make_tok_path(a.out_path, tok_path, sizeof(tok_path));
            tokenizer_save(tok, tok_path);
            printf("Saved checkpoint: %s (vocab: %s)\n", a.out_path, tok_path);
        } else {
            fprintf(stderr, "Note: no --out path; weights and tokenizer not saved.\n");
        }

        if (a.generate_seed) {
            printf("\n--- Sample (temperature %.2f) ---\n", a.temperature);
            text_lm_generate_with_tokenizer(model, &hp, tok, a.generate_seed, a.gen_length,
                                            a.temperature, stdout);
        }

        text_lm_free_session(model, ts);
        tokenizer_free(tok);
        return 0;
    }

    /* --load */
    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_load(a.load_path, &model, &ts) != 0 || !model) {
        fprintf(stderr, "Failed to load model: %s\n", a.load_path);
        return 1;
    }
    char tok_path[1024];
    make_tok_path(a.load_path, tok_path, sizeof(tok_path));
    Tokenizer *tok = tokenizer_load(tok_path);

    TextLmHyperparams hp;
    hyperparams_from_model(model, &hp);
    if (tok) hp.vocab_size = tok->vocab_size;

    printf("Loaded: %s (d_model=%d, layers enc/dec=%d/%d, vocab=%d%s)\n",
           a.load_path, model->config.d_model, model->config.encoder_layers,
           model->config.decoder_layers, hp.vocab_size, tok ? "" : ", no vocab file");

    if (a.generate_seed) {
        if (tok) {
            text_lm_generate_with_tokenizer(model, &hp, tok, a.generate_seed, a.gen_length,
                                            a.temperature, stdout);
        } else {
            text_lm_generate(model, &hp, a.generate_seed, a.gen_length, a.temperature, stdout);
        }
    } else {
        printf("No --generate; nothing else to do.\n");
    }

    text_lm_free_session(model, ts);
    if (tok) tokenizer_free(tok);
    return 0;
}
