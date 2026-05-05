#include "text_lm.h"
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char *read_text_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    if (out_len) {
        *out_len = n;
    }
    return buf;
}

static void hyperparams_from_model(const Transformer *m, TextLmHyperparams *hp) {
    *hp = text_lm_default_hyperparams;
    hp->d_model = m->config.d_model;
    hp->d_ff = m->config.d_ff;
    hp->nhead = m->config.nhead;
    hp->encoder_layers = m->config.encoder_layers;
    hp->decoder_layers = m->config.decoder_layers;
    hp->dropout = (float)m->config.dropout;
    if (m->config.max_len > 5) {
        hp->seq_len = m->config.max_len - 5;
    }
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --train <corpus.txt> [--out <model.bin>]\n"
            "  %s --load <model.bin> [--generate <seed>] [--length <n>] [--temperature <t>]\n"
            "\n"
            "  --train <path>       Train character LM on text file (a-z, space, dot).\n"
            "  --out <path>         Save weights after training (binary checkpoint).\n"
            "  --load <path>        Load checkpoint from --out.\n"
            "  --generate <str>     After --load, sample continuation (quote if spaces).\n"
            "  --length <n>         Generation length (default 60).\n"
            "  --temperature <t>    Sampling temperature (default 0.8).\n",
            argv0, argv0);
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *out_path = NULL;
    const char *load_path = NULL;
    const char *generate_seed = NULL;
    int gen_length = 60;
    float temperature = 0.8f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_path = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_seed = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--length") == 0 && i + 1 < argc) {
            gen_length = atoi(argv[++i]);
            if (gen_length < 1) {
                gen_length = 60;
            }
            continue;
        }
        if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)strtod(argv[++i], NULL);
            continue;
        }
        fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        return 1;
    }

    if (!train_path && !load_path) {
        print_usage(argv[0]);
        return 1;
    }
    if (train_path && load_path) {
        fprintf(stderr, "Use either --train or --load, not both.\n");
        return 1;
    }

    srand((unsigned)time(NULL));

    if (train_path) {
        size_t raw_len = 0;
        char *text = read_text_file(train_path, &raw_len);
        if (!text) {
            fprintf(stderr, "Cannot read training file: %s\n", train_path);
            return 1;
        }

        TextCorpus corpus = {0};
        if (text_corpus_from_text(text, &corpus) != 0) {
            fprintf(stderr, "No usable characters in corpus (expected a-z, space, dot).\n");
            free(text);
            return 1;
        }

        printf("=== Train from %s ===\n", train_path);
        printf("Vocab: %d chars (%s), hyperparams: seq=%d batch=%d d_model=%d\n", TEXT_LM_VOCAB_SIZE,
               text_lm_vocab_chars(), text_lm_default_hyperparams.seq_len,
               text_lm_default_hyperparams.batch_size, text_lm_default_hyperparams.d_model);
        printf("Corpus tokens: %d (from %zu raw bytes)\n\n", corpus.len, raw_len);

        Transformer *model = NULL;
        TrainingState *ts = NULL;
        int tr = text_lm_train(&corpus, &text_lm_default_hyperparams, &model, &ts);
        text_corpus_free(&corpus);
        free(text);

        if (tr != 0) {
            fprintf(stderr, "Training failed (%d). Corpus may be shorter than seq_len + batch.\n", tr);
            if (model) {
                text_lm_free_session(model, ts);
            }
            return 1;
        }

        printf("\n=== Training complete ===\n");

        if (out_path) {
            if (text_lm_save(model, ts, text_lm_default_hyperparams.learning_rate, out_path) != 0) {
                fprintf(stderr, "Failed to write model file: %s\n", out_path);
                text_lm_free_session(model, ts);
                return 1;
            }
            printf("Saved checkpoint: %s\n", out_path);
        } else {
            fprintf(stderr, "Note: no --out path; weights not saved.\n");
        }

        if (generate_seed) {
            TextLmHyperparams hp = text_lm_default_hyperparams;
            hyperparams_from_model(model, &hp);
            printf("\n--- Sample (temperature %.2f) ---\n", temperature);
            text_lm_generate(model, &hp, generate_seed, gen_length, temperature, stdout);
        }

        text_lm_free_session(model, ts);
        return 0;
    }

    /* --load */
    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_load(load_path, &model, &ts) != 0 || !model) {
        fprintf(stderr, "Failed to load model: %s\n", load_path);
        return 1;
    }

    TextLmHyperparams hp = text_lm_default_hyperparams;
    hyperparams_from_model(model, &hp);

    printf("Loaded: %s (d_model=%d, layers enc/dec=%d/%d)\n", load_path, model->config.d_model,
           model->config.encoder_layers, model->config.decoder_layers);

    if (generate_seed) {
        text_lm_generate(model, &hp, generate_seed, gen_length, temperature, stdout);
    } else {
        printf("No --generate; nothing else to do.\n");
    }

    text_lm_free_session(model, ts);
    return 0;
}
