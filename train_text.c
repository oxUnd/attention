#include "text_lm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char CORPUS[] =
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "the early bird catches the worm. "
    "a journey of a thousand miles begins with a single step. "
    "all that glitters is not gold. "
    "practice makes perfect. "
    "where there is a will there is a way. "
    "actions speak louder than words. "
    "knowledge is power. "
    "the pen is mightier than the sword. "
    "an apple a day keeps the doctor away. "
    "better late than never. "
    "curiosity killed the cat. "
    "every cloud has a silver lining. "
    "the only thing we have to fear is fear itself. "
    "to be or not to be that is the question. "
    "rome was not built in a day. "
    "when in rome do as the romans do. "
    "the grass is always greener on the other side. "
    "do not count your chickens before they hatch. ";

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== Transformer Character-Level Text Training (built-in corpus) ===\n");
    printf("Vocab: %d chars (%s), Seq: %d, Batch: %d, d_model: %d\n\n", TEXT_LM_VOCAB_SIZE,
           text_lm_vocab_chars(), text_lm_default_hyperparams.seq_len,
           text_lm_default_hyperparams.batch_size, text_lm_default_hyperparams.d_model);

    TextCorpus corpus = {0};
    if (text_corpus_from_text(CORPUS, &corpus) != 0) {
        fprintf(stderr, "Corpus preprocess failed.\n");
        return 1;
    }
    printf("Corpus: %d characters\n", corpus.len);
    printf("Sample: \"%.50s...\"\n\n", CORPUS);

    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_train(&corpus, &text_lm_default_hyperparams, &model, &ts) != 0) {
        text_corpus_free(&corpus);
        return 1;
    }
    text_corpus_free(&corpus);

    printf("\n=== Training Complete ===\n\n");

    printf("--- Temperature 0.5 (conservative) ---\n");
    text_lm_generate(model, &text_lm_default_hyperparams, "the ", 60, 0.5f, stdout);

    printf("--- Temperature 1.0 (balanced) ---\n");
    text_lm_generate(model, &text_lm_default_hyperparams, "the ", 60, 1.0f, stdout);

    printf("--- Temperature 1.5 (creative) ---\n");
    text_lm_generate(model, &text_lm_default_hyperparams, "the ", 60, 1.5f, stdout);

    printf("--- Different seed: \"she \" ---\n");
    text_lm_generate(model, &text_lm_default_hyperparams, "she ", 60, 0.8f, stdout);

    text_lm_free_session(model, ts);
    return 0;
}
